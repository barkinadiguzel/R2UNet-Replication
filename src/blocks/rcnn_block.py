from .rcl import RCL
import torch.nn as nn

class RCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RCNNBlock, self).__init__()
        self.rcl1 = RCL(in_channels, out_channels, t)
        self.rcl2 = RCL(out_channels, out_channels, t)

    def forward(self, x):
        out = self.rcl1(x)
        out = self.rcl2(out)
        return out
