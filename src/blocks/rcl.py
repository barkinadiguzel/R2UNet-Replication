import torch
import torch.nn as nn
import torch.nn.functional as F

class RCL(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RCL, self).__init__()
        self.t = t
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.bn(F.relu(self.conv_in(x)))
        out = x1
        for _ in range(self.t):
            out = self.bn(F.relu(self.conv(out) + x1))
        return out
