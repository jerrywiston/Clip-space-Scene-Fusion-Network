import torch
import torch.nn as nn
from padding_same_conv import Conv2d

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGRUCell, self).__init__()
        kwargs = dict(kernel_size=kernel_size, stride=stride)
        in_channels += out_channels
        
        self.reset_conv = Conv2d(in_channels, out_channels, **kwargs)
        self.update_conv = Conv2d(in_channels, out_channels, **kwargs)
        self.state_conv = Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, state):
        input_cat1 = torch.cat((state, input), dim=1)
        reset_gate = torch.sigmoid(self.reset_conv(input_cat1))
        update_gate = torch.sigmoid(self.update_conv(input_cat1))

        state_reset = reset_gate * state
        input_cat2 = torch.cat((state_reset, input), dim=1)
        state_update = (1-update_gate)*state + update_gate*torch.tanh(self.state_conv(input_cat2))
        return state_update 