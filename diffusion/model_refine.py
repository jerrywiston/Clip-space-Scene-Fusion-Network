import torch
import torch.nn as nn
import math
from modules import *


class DiffusionModel(nn.Module):
    def __init__(self, in_size, t_range, img_depth, cell_size, device):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        self.cell_size = cell_size
        self.device = device

        bilinear = True
        self.inc = DoubleConv(img_depth*2, 64)
        self.down1 = Down(64+cell_size, 128) #(64,64)->(32,32)
        self.down2 = Down(128+cell_size, 256) #(32,32)->(16,16)
        factor = 2 if bilinear else 1
        self.down3 = Down(256+cell_size, 512 // factor) #(16,16)->(8,8)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256+cell_size, 128 // factor, bilinear)
        self.up3 = Up(128+cell_size, 64, bilinear)
        self.outc = OutConv(64+cell_size, img_depth)
        
        img_size = int(self.in_size ** (1/2))
        self.sa1 = SAWrapper(256, int(img_size/4))
        self.sa2 = SAWrapper(256, int(img_size/8))
        self.sa3 = SAWrapper(128, int(img_size/4))

        self.up2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        
    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, xd, c, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        img_size = int(self.in_size ** (1/2))

        # Down
        x1 = self.inc(torch.cat((x, xd), 1))
        x1_ = torch.cat((x1, self.up4x4(c)), 1)

        x2 = self.down1(x1_) + self.pos_encoding(t, 128, int(img_size/2))
        x2_ = torch.cat((x2, self.up2x2(c)), 1)
        
        x3 = self.down2(x2_) + self.pos_encoding(t, 256, int(img_size/4))
        x3 = self.sa1(x3)
        x3_ = torch.cat((x3, c), 1)
        
        x4 = self.down3(x3_) + self.pos_encoding(t, 256, int(img_size/8))
        x4 = self.sa2(x4)
        
        # Up
        x5 = self.up1(x4, x3) + self.pos_encoding(t, 128, int(img_size/4))
        x5 = self.sa3(x5)
        x5_ = torch.cat((x5, c), 1)

        x6 = self.up2(x5_, x2) + self.pos_encoding(t, 64, int(img_size/2))
        x6_ = torch.cat((x6, self.up2x2(c)), 1)

        x7 = self.up3(x6_, x1) + self.pos_encoding(t, 64, int(img_size))
        x7_ = torch.cat((x7, self.up4x4(c)), 1)
        output = self.outc(x7_)
        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, x, xd, c):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [x.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(x.shape, device=self.device)
        
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * x[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, xd, c, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, xd, c, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(self.device)
            else:
                z = 0
            e_hat = self.forward(x, xd, c, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x
    
    def denoise_sample_ddim(self, x, xd, c, t, samp_step, eta=0.0):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(self.device)
            else:
                z = 0
            e_hat = self.forward(x, xd, c, t.view(1, 1).repeat(x.shape[0], 1))
            if t-1 == 0:
                alpha_past = 1
            else:
                alpha_past = self.alpha_bar(t-samp_step)
            alpha = self.alpha_bar(t)
            term1 = math.sqrt(alpha_past) * (x-math.sqrt(1-alpha)*e_hat) / math.sqrt(alpha)
            sigma = eta*math.sqrt((1-alpha_past)/(1-alpha))*math.sqrt(1-alpha/alpha_past)
            term2 = math.sqrt(1-alpha_past-sigma**2) * e_hat
            term3 = sigma * z
            x = term1 + term2 + term3
            return x

    def image_sample(self, xd, c, batch_size):
        x = torch.randn((batch_size, 3, xd.shape[-2], xd.shape[-1])).to(self.device)
        sample_steps = torch.arange(self.t_range-1, 1, -1).to(self.device)
        for t in sample_steps:
            x = self.denoise_sample(x, xd, c, t)
        #x = (x.clamp(-1,1) + 1) / 2
        x = x.clamp(0,1)
        return x

    def image_sample_ddim(self, xd, c, batch_size):
        x = torch.randn((batch_size, 3, 64, 64)).to(self.device)
        samp_step = 10
        sample_steps = torch.arange(self.t_range-1, 0, -samp_step).to(self.device)
        print(sample_steps)
        for t in sample_steps:
            print(t)
            x = self.denoise_sample_ddim(x, xd, c, t, samp_step)
        #x = (x.clamp(-1,1) + 1) / 2
        x = x.clamp(0,1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
