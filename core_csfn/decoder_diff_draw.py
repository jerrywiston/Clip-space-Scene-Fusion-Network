import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d
import conv_draw
import numpy as np
from diffusion import model_refine as model

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
        diffusion_steps = 1000
        self.dec_draw = conv_draw.GeneratorNetwork(x_dim=3, r_dim=cell_size, L=6, scale=4, share=True).to(device)
        self.dec_diff = model.DiffusionModel(64*64, diffusion_steps, 3, cell_size, device).to(device)

    def draw_canvas(self, view_cell_q, render_layers=-1):
        if render_layers == -1:
            depth_size = view_cell_q.shape[2]
        else:
            depth_size = render_layers
        vshape = view_cell_q.shape
        canvas = view_cell_q[:,:,0,:,:]
        for i in range(1,depth_size):
            mask = self.mask_cnn(canvas)
            canvas = mask*view_cell_q[:,:,i,:,:] + (1-mask)*canvas
        return canvas

    def get_loss(self, x, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q) 

        # DRAW
        x_rec, kl = self.dec_draw(x, canvas)
        kl_loss = torch.mean(torch.sum(kl, dim=[1,2,3]))
        rec_loss = nn.MSELoss()(x_rec, x)
        draw_loss = rec_loss + 0.001*kl_loss
                             
        # Diffusion Refinement
        diff_loss = self.dec_diff.get_loss(x, x_rec.detach(), canvas)
    
        return kl_loss, rec_loss, draw_loss, diff_loss, x_rec

    def sample(self, view_cell_q, render_layers=-1):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q, render_layers)
        x_samp_draw = self.dec_draw.sample((64,64), canvas)
        x_samp_diff = self.dec_diff.image_sample(x_samp_draw, canvas, canvas.shape[0])
        return x_samp_draw, x_samp_diff
