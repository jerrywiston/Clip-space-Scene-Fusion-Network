import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d
import conv_gru

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderNetworkRes(nn.Module):
    def __init__(self, ch=64, csize=64):
        super(DecoderNetworkRes, self).__init__()
        # Stem
        self.conv1 = Conv2d(csize, ch*2, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(ch*2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # ResBlock 1
        self.bn2 = nn.BatchNorm2d(ch*2)
        self.conv2 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(ch*2)
        self.conv3 = Conv2d(ch*2, ch*2, 3, stride=1)
        # Pool + Feature Dim
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = Conv2d(ch*2, ch, 1, stride=1)
        # ResBlock 2
        self.bn5 = nn.BatchNorm2d(ch)
        self.conv5 = Conv2d(ch, ch, 3, stride=1)
        self.bn6 = nn.BatchNorm2d(ch)
        self.conv6 = Conv2d(ch, ch, 3, stride=1)
        # Feature Dim
        self.bn7 = nn.BatchNorm2d(ch)
        self.conv7 = Conv2d(ch, 3, 3, stride=1)

    def forward(self, f):
        # Stem
        out = F.leaky_relu(self.bn1(self.conv1(f)))
        out = self.up1(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(self.bn2(out), inplace=True))
        out = self.conv3(F.leaky_relu(self.bn3(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.up3(out)
        out = self.conv4(out)
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(self.bn5(out), inplace=True))
        out = self.conv6(F.leaky_relu(self.bn6(hidden), inplace=True)) + out
        # Feature Dim
        out = self.conv7(self.bn7(out))
        return torch.sigmoid(out)

class RendererGRU(nn.Module):
    def __init__(self, cell_size):
        super(RendererGRU, self).__init__()
        self.cell_size = cell_size
        self.cgru = conv_gru.ConvGRUCell(cell_size, cell_size).to(device)
        self.dec = DecoderNetworkRes(csize=cell_size).to(device)

    def forward(self, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        depth_size = view_cell_q.shape[2]
        vshape = view_cell_q.shape
        state_update = torch.zeros((vshape[0], vshape[1], vshape[3], vshape[4])).to(device) # init
        for i in range(depth_size):
            state_update = self.cgru(view_cell_q[:,:,i,:,:], state_update)
        xq = self.dec(state_update)
        
        return xq

class RendererMask(nn.Module):
    def __init__(self, cell_size):
        super(RendererMask, self).__init__()
        self.mask_cnn = nn.Sequential(
            Conv2d(cell_size, 64, 3, stride=1),
            nn.ReLU(),
            Conv2d(64, 1, 3, stride=1),
            nn.Sigmoid(),
        )
        self.dec = DecoderNetworkRes(csize=cell_size).to(device)

    def forward(self, view_cell_q):
        # view_cell_q: (b,c,d,h,w)
        depth_size = view_cell_q.shape[2]
        vshape = view_cell_q.shape
        canvas = view_cell_q[:,:,0,:,:]
        for i in range(1,depth_size):
            mask = self.mask_cnn(canvas)
            canvas = mask*view_cell_q[:,:,i,:,:] + (1-mask)*canvas

        xq = self.dec(canvas)
        
        return xq
