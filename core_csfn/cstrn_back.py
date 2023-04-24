import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_quaterion = True
use_pos_encoding = True

# Clip Space Spatial Transformation Routing Network
class CSTRN(nn.Module):
    def __init__(self, view_size=(16,16), depth_size=6, pose_size=12, emb_size=32, cell_size=128, fusion_type="ocm"):
        super(CSTRN, self).__init__()
        self.view_size = view_size # View Space Size
        self.depth_size = depth_size # Dpeth Layer Size
        self.pose_size = pose_size # Pose Size
        self.emb_size = emb_size # Embedding Size
        self.cell_size = cell_size # Cell Size
        self.fusion_type = fusion_type # "ocm"/"sum"/"avg"/"norm"

        if use_quaterion:
            self.pose_size = 7

        # 2D Observation View Space Embedding Network
        self.emb_net_k = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb_size)
        ).to(device)

        # 2.5D Query View Space Embedding Network
        self.emb_net_q = nn.Sequential(
            nn.Linear(3 + self.pose_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_size),
        ).to(device)

        # Transformation Routing Network
        self.act_net = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(device)
        #self.attn_net = nn.MultiheadAttention(embed_dim=cell_size, num_heads=1, batch_first=True).to(device)

    def forward(self, view_cell, pose_o, pose_q):
        # pose_o: (b,n,12)
        # pose_q: (b,1,12)
        # query_view_cell: (b*n,c,h,w)

        batch_size = pose_o.shape[0]
        num_obs = pose_o.shape[1]

        if use_quaterion:
            pose_o_quat = self.mat2quat(pose_o)
            pose_q_quat = self.mat2quat(pose_q)

        # Observation View Space Embedding
        ho = torch.linspace(-1, 1, self.view_size[0])   #(h)
        wo = torch.linspace(-1, 1, self.view_size[1])   #(w)
        ho_grid, wo_grid = torch.meshgrid(ho, wo)   
        code_o = torch.cat((ho_grid.unsqueeze(-1), wo_grid.unsqueeze(-1)), dim=-1).reshape(-1,2).to(device) #(h*w, 2)
        emb_o = self.emb_net_k(code_o)  #(h*w,e)
        emb_o = torch.unsqueeze(emb_o, 0).repeat(pose_o.shape[0]*pose_o.shape[1], 1, 1) #(b*n, h*w, e)

        # Query View Space Embedding
        hq = torch.linspace(-1, 1, self.view_size[0])   #(h)
        wq = torch.linspace(-1, 1, self.view_size[1])   #(w)
        dq = torch.linspace(0, 1, self.depth_size)      #(d)
        dq_grid, hq_grid, wq_grid = torch.meshgrid(dq, hq, wq)
        code_q = torch.cat((dq_grid.unsqueeze(-1), hq_grid.unsqueeze(-1), wq_grid.unsqueeze(-1)), dim=-1).reshape(-1,3).to(device) #(d*h*w, 3)
        code_q = torch.unsqueeze(code_q, 0).repeat(pose_o.shape[0]*pose_o.shape[1], 1, 1) #(b*n,d*h*w,3)
        pose_trans = self.transform_pose(pose_o, pose_q).reshape(-1,12)    #(b*n,12)
        pose_trans_tile = pose_trans.unsqueeze(1).repeat(1,self.depth_size*self.view_size[0]*self.view_size[1],1)  #(b*n,d*h*w,12)
        emb_q = self.emb_net_q(torch.cat((pose_trans_tile, code_q), 2))    #(b*n,d*h*w,15)
        mask = self.act_net(emb_q.reshape(-1,self.emb_size)).reshape(-1,self.depth_size*self.view_size[0]*self.view_size[1],1)    #(b*n,d*h*w,1)

        # Routing
        view_cell_permute = view_cell.permute(0,2,3,1).reshape(-1,view_cell.shape[2]*view_cell.shape[3],view_cell.shape[1])
        attn_output = self.attention(emb_q, emb_o, view_cell_permute) # Q, K, V (b*n,d*h*w,c)
        query_view_cell_expend = mask * attn_output # (b*n,d*h*w,c)
        query_view_cell = self.fusion(query_view_cell_expend, batch_size, num_obs) # (b,d*h*w,c)
        query_view_cell = query_view_cell.permute(0,2,1) # (b,c,d*h*w)
        query_view_cell = query_view_cell.reshape(-1, self.cell_size, self.depth_size, self.view_size[0], self.view_size[1])
        # (b,c,d,h,w)
        return query_view_cell
    
    def transform_pose(self, pose_o, pose_q):
        # pose_o: (b,n,12)
        # pose_q: (b,1,12)

        if use_quaterion:
            pose_o = self.mat2quat(pose_o)
            pose_q = self.mat2quat(pose_q)

        batch_size = pose_o.shape[0]
        num_obs = pose_o.shape[1]
        with torch.no_grad():
            vec_affine = torch.tensor([0,0,0,1]).to(device)
            
            v_aff_o = vec_affine.reshape([1,1,4]).repeat(batch_size, num_obs, 1)
            mtx_o = torch.cat([pose_o, v_aff_o],2).reshape(batch_size, num_obs, 4, 4)  # (b,n,4,4)
            
            v_aff_q = vec_affine.reshape([1,1,4]).repeat(batch_size, 1, 1)
            mtx_q = torch.cat([pose_q, v_aff_q],2).reshape(batch_size, 1, 4, 4).repeat(1,num_obs,1,1)  # (b,n,4,4)
            
            mtx_o_inv = torch.linalg.inv(mtx_o.reshape(-1,4,4)).reshape(mtx_o.shape)
            mtx_q_trans = torch.matmul(mtx_o_inv, mtx_q)  # (b,n,4,4)

            pose_q_trans = mtx_q_trans.reshape(batch_size, num_obs, 16)[...,:12]    # (b,n,12)
        return pose_q_trans
    
    def attention(self, q, k, v):
        # Q: (b*n, d*h*w, e), K: (b*n, h*w, e), V: (b*n, h*w, c)
        rmtx = torch.matmul(q, k.permute(0,2,1)) # (b*n, d*h*w, h*w)
        att = torch.softmax(rmtx, 2) 
        vatt = torch.matmul(att, v)
        return vatt

    def fusion(self, query_view_cell_exp, batch_size, num_obs): # (b*n,d*h*w,c)
        query_view_cell = query_view_cell_exp.reshape(batch_size, num_obs, -1, self.cell_size)
        
        if self.fusion_type == "ocm":
            query_view_cell = torch.sum(query_view_cell, 1, keepdim=False)
            query_view_cell = torch.sigmoid(query_view_cell)
        elif self.fusion_type == "sum":
            query_view_cell = torch.sum(query_view_cell, 1, keepdim=False)
        elif self.fusion_type == "avg":
            query_view_cell = torch.mean(query_view_cell, 1, keepdim=False)
        elif self.fusion_type == "norm":
            query_view_cell = torch.sum(query_view_cell, 1, keepdim=False)
            fusion_norm = torch.norm(query_view_cell, -1, keepdim=True)
            query_view_cell = query_view_cell / fusion_norm

        return query_view_cell
    
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    def mat2quat(self, tmat):
        input_shape = tmat.shape
        output_shape = list(torch.tensor(tmat.shape))
        output_shape[-1] = 7
        tmat_re = tmat.reshape(-1,input_shape[-1])

        temp = []
        for i in range(tmat_re.shape[0]):
            trans = torch.FloatTensor([tmat_re[i,3], tmat_re[i,7], tmat_re[i,11]]).to(device)
            
            m00 = tmat_re[i,0].unsqueeze(0)
            m01 = tmat_re[i,1].unsqueeze(0)
            m02 = tmat_re[i,2].unsqueeze(0)
            m10 = tmat_re[i,4].unsqueeze(0)
            m11 = tmat_re[i,5].unsqueeze(0)
            m12 = tmat_re[i,6].unsqueeze(0)
            m20 = tmat_re[i,8].unsqueeze(0)
            m21 = tmat_re[i,9].unsqueeze(0)
            m22 = tmat_re[i,10].unsqueeze(0)
            if m22 < 0:
                if m00 > m11:
                    t = 1 + m00 - m11 - m22
                    q = torch.cat([t, m01+m10, m20+m02, m12-m21], 0)
                else:
                    t = 1 - m00 + m11 - m22
                    q = torch.cat([m01+m10, t, m12+m21, m20-m02], 0)
            else:
                if m00 < -m11:
                    t = 1 - m00 - m11 + m22
                    q = torch.cat([m20+m02, m12+m21, t, m01-m10], 0)
                else:
                    t = 1 + m00 + m11 + m22
                    q = torch.cat([m12-m21, m20-m02, m01-m10, t], 0)

            q = q * (0.5 / torch.sqrt(t))
            q = torch.cat([trans, q], 0)
            temp.append(q)
        temp = torch.cat(temp, 0).reshape(output_shape)
        return temp