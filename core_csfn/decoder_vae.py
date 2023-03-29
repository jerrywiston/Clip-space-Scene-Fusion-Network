import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
from blurPooling import BlurPool2d
import conv_gru
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
        self.dec_vae_encoder = CvaeEncoder(z_dim=z_dim, c_dim=cell_size).to(device)
        self.dec_vae_decoder = CvaeDecoder(z_dim=z_dim, c_dim=cell_size).to(device)

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
        z_samp, z_mu, z_logvar = self.dec_vae_encoder(x, canvas)
        x_rec = self.dec_vae_decoder(z_samp, canvas)
        
        # VAE Loss
        rec_loss = nn.MSELoss()(x_rec, x)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1))
        loss = rec_loss + 0.01*kl_loss #0.0001
        return x_rec, loss, kl_loss, rec_loss

    def sample(self, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        canvas = self.draw_canvas(view_cell_q)
        z_samp = torch.randn(canvas.shape[0], self.z_dim, device=device)
        x_samp = self.dec_vae_decoder(z_samp, canvas)
        return x_samp
    
# ===========================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def NormalNLLLoss(x, mu, logvar):
    NormalNLL =  0.5 * (np.log(2*np.pi) + logvar) + (x - mu)**2 / (2*torch.exp(logvar))
    return NormalNLL

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn=True, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn
        self.sn = sn
        if sn:
            self.conv0 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        h_conv0 = self.conv0(x)
        if self.bn:
            h_conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
            h_conv2 = self.bn2(self.conv2(h_conv1))
        else:
            h_conv1 = F.relu(self.conv1(x), inplace=True)
            h_conv2 = self.conv2(h_conv1)
        out = F.relu(h_conv0 + h_conv2, inplace=True)
        return out

class CvaeEncoder(nn.Module):
    def __init__(self, z_dim, c_dim, ndf=128):
        super(CvaeEncoder, self).__init__()
        self.ndf = ndf
        # (64,64,3) -> (32,32,128)
        self.res1 = ResBlock(3, ndf, sn=False)
        self.pool1 = BlurPool2d(filt_size=3, channels=ndf, stride=2)
        # (32,32,128) -> (16,16,256)
        self.res2 = ResBlock(ndf, ndf*2, sn=False)
        self.pool2 = BlurPool2d(filt_size=3, channels=ndf*2, stride=2)
        # (16,16,256) -> (8,8,512)
        self.res3 = ResBlock(ndf*2 + c_dim, ndf*4, sn=False)
        self.pool3 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (8,8,512) -> (4,4,512)
        self.res4 = ResBlock(ndf*4, ndf*4, sn=False)
        self.pool4 = BlurPool2d(filt_size=3, channels=ndf*4, stride=2)
        # (4*4*512 -> z_dim)
        self.fc5_mu = nn.Linear(4*4*ndf*4, z_dim)
        self.fc5_logvar = nn.Linear(4*4*ndf*4, z_dim)
    
    def forward(self, x, c):
        # Res Block
        h_res1 = self.res1(x)
        h_pool1 = self.pool1(h_res1)
        h_res2 = self.res2(h_pool1)
        h_pool2 = self.pool2(h_res2)
        h_pool2 = torch.cat((h_pool2, c), 1)
        h_res3 = self.res3(h_pool2)
        h_pool3 = self.pool3(h_res3)
        h_res4 = self.res4(h_pool3)
        h_pool4 = self.pool4(h_res4)
        # Fully Connected
        z_mu = self.fc5_mu(h_pool4.view(-1,self.ndf*4*4*4))
        z_logvar = self.fc5_logvar(h_pool4.view(-1,self.ndf*4*4*4))
        z_samp = self.sample_z(z_mu, z_logvar)
        return z_samp, z_mu, z_logvar
    
    def sample_z(self, mu, logvar):
        eps = torch.randn(mu.size()).to(device)
        return mu + torch.exp(logvar / 2) * eps

class CvaeDecoder(nn.Module):
    def __init__(self, z_dim, c_dim, ndf=64):
        super(CvaeDecoder, self).__init__()
        self.ndf = ndf
        # (8,8,512) -> (8,8,512)
        self.fc1 = nn.Linear(z_dim, 8*8*ndf*8)
        self.res1 = ResBlock(ndf*8,ndf*8, sn=False)
        # (8,8,512) -> (16,16,256)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res2 = ResBlock(ndf*8,ndf*4, sn=False)
        # (16,16,256) -> (32,32,256)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res3 = ResBlock(ndf*4+c_dim,ndf*4, sn=False)
        # (32,32,256) -> (64,64,128)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.res4 = ResBlock(ndf*4,ndf*2, sn=False)
        # (64,64,64) -> (64,64,3)
        self.conv5 = nn.Conv2d(in_channels=ndf*2, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, c):
        h_fc1 = F.relu(self.fc1(z).view(-1,self.ndf*8,8,8)) 
        h_res1 = self.res1(h_fc1)
        h_up2 = self.up2(h_res1)
        h_res2 = self.res2(h_up2)
        h_res2 = torch.cat((h_res2, c), 1)
        h_up3 = self.up3(h_res2)
        h_res3 = self.res3(h_up3)
        h_up4 = self.up4(h_res3)
        h_res4 = self.res4(h_up4)
        x_samp = torch.sigmoid(self.conv5(h_res4))
        return x_samp
