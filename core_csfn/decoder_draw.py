import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d
import conv_draw
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Renderer(nn.Module):
    def __init__(self, cell_size, z_dim):
        super(Renderer, self).__init__()
        self.cell_size = cell_size
        self.z_dim = z_dim
        self.mask_cnn = nn.Sequential(
            Conv2d(cell_size, 64, 3, stride=1),
            nn.ReLU(),
            Conv2d(64, 1, 3, stride=1),
            nn.Sigmoid(),
        )

        self.dec = conv_draw.GeneratorNetwork(x_dim=3, r_dim=cell_size, L=6, scale=4, share=True).to(device)

    def draw_canvas(self, view_cell_q):
        depth_size = view_cell_q.shape[2]
        vshape = view_cell_q.shape
        canvas = view_cell_q[:,:,0,:,:]
        for i in range(1,depth_size):
            mask = self.mask_cnn(canvas)
            canvas = mask*view_cell_q[:,:,i,:,:] + (1-mask)*canvas
        return canvas

    def forward(self, x, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q) 

        # Inference
        x_rec, kl = self.dec(x, canvas)
        kl_loss = torch.mean(torch.sum(kl, dim=[1,2,3]))

        # VAE Loss
        rec_loss = nn.MSELoss()(x_rec, x)
        loss = rec_loss + 0.001*kl_loss #0.0001
        return x_rec, loss, kl_loss, rec_loss

    def sample(self, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q)
        x_samp = self.dec.sample((64,64), canvas)
        return x_samp
