import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
#import decoder_vae as decoder
import decoder_draw as decoder
import cstrn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clip Space Scene Fusion Network
class CSFN(nn.Module):
    def __init__(self, view_size=(16,16), depth_size=6, pose_size=12, emb_size=32, cell_size=128, fusion_type="ocm"):
        super(CSFN, self).__init__()
        # Set Variables
        self.view_size = view_size
        self.depth_size = depth_size
        self.pose_size = pose_size
        self.emb_size = emb_size
        self.cell_size = cell_size
        self.fusion_type = fusion_type

        # Build Modules
        self.encoder = encoder.EncoderNetworkRes(csize=cell_size).to(device)
        self.cstrn = cstrn.CSTRN(view_size, depth_size, pose_size, emb_size, cell_size, fusion_type)
        self.renderer = decoder.Renderer(cell_size=cell_size, z_dim=64).to(device)

    def forward(self, xo, xq, pose_o, pose_q):
        view_cell_o = self.encoder(xo)
        view_cell_q_layers = self.cstrn(view_cell_o, pose_o, pose_q)
        xq_rec, loss, kl_loss, rec_loss = self.renderer(xq, view_cell_q_layers)
        return xq_rec, loss, kl_loss, rec_loss
    
    def sample(self, xo, pose_o, pose_q):
        view_cell_o = self.encoder(xo)
        view_cell_q_layers = self.cstrn(view_cell_o, pose_o, pose_q)
        xq_samp = self.renderer.sample(view_cell_q_layers)
        return xq_samp
