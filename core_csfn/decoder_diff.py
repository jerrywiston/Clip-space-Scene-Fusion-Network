import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d
import conv_gru
import numpy as np
from diffusion import model

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
        self.dec_diff = model.DiffusionModel(64*64, diffusion_steps, 3, cell_size, device).to(device)

    def draw_canvas(self, view_cell_q):
        depth_size = view_cell_q.shape[2]
        vshape = view_cell_q.shape
        canvas = view_cell_q[:,:,0,:,:]
        for i in range(1,depth_size):
            mask = self.mask_cnn(canvas)
            canvas = mask*view_cell_q[:,:,i,:,:] + (1-mask)*canvas
        return canvas

    def get_loss(self, x, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q) 

        # Inference
        loss = self.dec_diff.get_loss(x, canvas)
    
        return loss

    def sample(self, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q)

        x_samp = self.dec_diff.image_sample(canvas, canvas.shape[0])
        return x_samp
