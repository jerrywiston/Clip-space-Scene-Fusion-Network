import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderNetworkLight(nn.Module):
    def __init__(self, ch=64, csize=128):
        super(EncoderNetworkLight, self).__init__()
        self.net = nn.Sequential(
                # (ch,64,64)
                Conv2d(3, ch, 5, stride=2),
                nn.LeakyReLU(),
                # (ch,32,32)
                Conv2d(ch, ch, 5, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch),
                # (ch/2,32,32)
                Conv2d(ch, ch*2, 3, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(ch*2),
                # (ch/2,16,16)
                Conv2d(ch*2, csize, 3, stride=1),
                # (tsize,16,16)
        )
        
    def forward(self, x):
        out = self.net(x)
        return out

from blurPooling import BlurPool2d
class EncoderNetworkRes(nn.Module):
    def __init__(self, ch=64, csize=128, down_size=4):
        super(EncoderNetworkRes, self).__init__()
        # Stem
        self.conv1 = Conv2d(3, ch, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(ch)
        #self.pool1 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        # ResBlock 1
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = Conv2d(ch, ch, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(ch)
        self.conv3 = Conv2d(ch, ch, 3, stride=1)
        # Pool + Feature Dim
        #self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = BlurPool2d(filt_size=3, channels=ch, stride=2)
        self.conv4 = Conv2d(ch, ch*2, 1, stride=1)
        # ResBlock 2
        self.bn5 = nn.BatchNorm2d(ch*2)
        self.conv5 = Conv2d(ch*2, ch*2, 3, stride=1)
        self.bn6 = nn.BatchNorm2d(ch*2)
        self.conv6 = Conv2d(ch*2, ch*2, 3, stride=1)
        # Feature Dim
        self.bn7 = nn.BatchNorm2d(ch*2)
        self.conv7 = Conv2d(ch*2, csize, 1, stride=1)

    def forward(self, x):
        # Stem
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        #out = self.pool1(out)
        # ResBlock 1
        hidden = self.conv2(F.leaky_relu(self.bn2(out), inplace=True))
        out = self.conv3(F.leaky_relu(self.bn3(hidden), inplace=True)) + out
        # Pool + Feature Dim
        out = self.pool3(out)
        out = self.conv4(out)
        # ResBlock 2
        hidden = self.conv5(F.leaky_relu(self.bn5(out), inplace=True))
        out = self.conv6(F.leaky_relu(self.bn6(hidden), inplace=True)) + out
        # Feature Dim
        out = self.conv7(self.bn7(out))
        return out

class EncoderNetworkSA(nn.Module):
    def __init__(self, ch=64, csize=128, down_size=4, img_size=(64,64)):
        super(EncoderNetworkSA, self).__init__()
        self.img_size = img_size
        self.conv1 = DoubleConv(3, ch)
        self.down2 = Down(ch, ch*2) #(64,64)->(32,32)
        self.down3 = Down(ch*2, ch*4) #(32,32)->(16,16)
        self.sa4 = SAWrapper(ch*4, int(img_size[0]/down_size))
        self.conv5 = Conv2d(ch*4, csize, 3, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2) 
        x3 = self.sa4(x3)
        x4 = self.conv5(x3)
        return x4

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, 4, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SAWrapper(nn.Module):
    def __init__(self, h_size, num_s):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.num_s = num_s
        self.h_size = h_size

    def forward(self, x):
        x = x.view(-1, self.h_size, self.num_s * self.num_s).swapaxes(1, 2)
        x = self.sa(x)
        x = x.swapaxes(2, 1).view(-1, self.h_size, self.num_s, self.num_s)
        return x