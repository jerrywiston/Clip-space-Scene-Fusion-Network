import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryController(nn.Module):
    def __init__(self, n_wrd_cells, view_size=(16,16), vsize=12, emb_size=32, csize=128, use_pos_encoding=True):
        super(MemoryController, self).__init__()
        self.n_wrd_cells = n_wrd_cells
        self.view_size = view_size
        self.vsize = vsize
        self.emb_size = emb_size
        self.csize = csize
        self.use_pos_encoding = use_pos_encoding

        # Camera Space Embedding / Frustum Activation / Occlusion
        if use_pos_encoding:
            self.fc1 = nn.Linear(3*(6*2+1), 512)
        else:
            self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_act = nn.Linear(256, 1)
        self.fc_emb = nn.Linear(256, emb_size)

        # View Space Embedding Network
        if use_pos_encoding:
            self.vse = nn.Sequential(
                nn.Linear(2*(6*2+1), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, emb_size)
            )
        else:
            self.vse = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, emb_size)
            )

    def wcode2cam(self, v, wcode):
        #with torch.no_grad():
        # Transformation
        vec_affine = torch.tensor([0.,0.,0.,1.]).to(device)
        vec_affine = vec_affine.unsqueeze(0).repeat((v.shape[0],1))
        v_mtx = torch.cat((v, vec_affine),1).reshape(-1, 4, 4)
        v_mtx_inv = torch.linalg.inv(v_mtx)
        v_mtx_inv_tile = torch.unsqueeze(v_mtx_inv, 1).repeat(1, self.n_wrd_cells, 1, 1)

        wcode = torch.cat([wcode, torch.ones_like(wcode[:,:1])], 1)
        wcode_batch = torch.unsqueeze(wcode, 0).repeat(v.shape[0], 1, 1)

        wcode_batch_trans = torch.matmul(v_mtx_inv_tile, torch.unsqueeze(wcode_batch, 3))
        wcode_batch_trans = (wcode_batch_trans[...,0])[...,:3]

        return wcode_batch_trans

    def transform(self, wcode_batch_trans, view_size=None):
        if view_size is None:
            view_size = self.view_size

        # Get Transform Location Code of World Cells
        if self.use_pos_encoding:
            wcode_batch_trans = self.positional_encoding(wcode_batch_trans)
        h = F.relu(self.fc1(wcode_batch_trans.reshape(self.n_wrd_cells*wcode_batch_trans.shape[0],-1)))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        activation = torch.sigmoid(self.fc_act(h).view(-1, self.n_wrd_cells, 1))
        cs_embedding = self.fc_emb(h).view(-1, self.n_wrd_cells, self.emb_size)
        
        # View Space Embedding
        x = torch.linspace(-1, 1, view_size[0])
        y = torch.linspace(-1, 1, view_size[1])
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        vcode = torch.cat((torch.unsqueeze(x_grid, 0), torch.unsqueeze(y_grid, 0)), dim=0).reshape(2,-1).permute(1,0).to(device) #(16*16, 2)
        if self.use_pos_encoding:
            vcode = self.positional_encoding(vcode)
        vs_embedding = self.vse(vcode) #(256, 128)
        vs_embedding = torch.unsqueeze(vs_embedding, 0).repeat(wcode_batch_trans.shape[0], 1, 1) #(-1, view_cell, emb_size)
        
        # Cross-Space Cell Relation
        relation = torch.bmm(cs_embedding, vs_embedding.permute(0,2,1)) #(-1, wrd_cell, view_cell)
        return relation, activation

    def get_clip_tensor(self, wcode_batch, clip_range):
        with torch.no_grad():
            clip1 = (wcode_batch[:,:,0]<clip_range).float()
            clip2 = (wcode_batch[:,:,0]>-clip_range).float()
            clip3 = (wcode_batch[:,:,1]<clip_range).float()
            clip4 = (wcode_batch[:,:,1]>-clip_range).float()
            w_clip = clip1 * clip2 * clip3 * clip4
        return w_clip

    def forward(self, view_cell, v, wcode, view_size=None, clip_range=3):
        if view_size is None:
            view_size = self.view_size
        wcode_batch_trans = self.wcode2cam(v, wcode)
        relation, activation = self.transform(wcode_batch_trans, view_size=view_size)

        distribution = torch.softmax(relation, 2)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        wrd_cell = torch.bmm(view_cell, route.permute(0,2,1))

        # Clip the out-of-range cells.
        if clip_range is not None:
            w_clip = self.get_clip_tensor(wcode_batch_trans, clip_range)
            w_mask = w_clip.float().unsqueeze(1)
            wrd_cell = wrd_cell * w_mask

        return wrd_cell # (-1, csize, n_wrd_cells)
    
    def query(self, wrd_cell, v, wcode, view_size=None):
        if view_size is None:
            view_size = self.view_size

        wcode_batch_trans = self.wcode2cam(v, wcode)
        relation, activation = self.transform(wcode_batch_trans, view_size=view_size)

        distribution = torch.softmax(relation, 1)
        route = distribution * activation   # (-1, n_wrd_cells, n_view_cells)
        query_view_cell = torch.bmm(wrd_cell, route).reshape(-1, self.csize, view_size[0], view_size[1])
        return query_view_cell

    # https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX
    def positional_encoding(
        self, tensor, num_encoding_functions=6, include_input=True, log_sampling=True
    ) -> torch.Tensor:

        # TESTED
        # Trivially, the input tensor is added to the positional encoding.
        encoding = [tensor] if include_input else []
        # Now, encode the input using a set of high-frequency functions and append the
        # resulting values to the encoding.
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    num_encoding_functions - 1,
                    num_encoding_functions,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)