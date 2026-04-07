import torch.nn as nn
from .rcnn_block import RCNNBlock

class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNNBlock, self).__init__()
        self.rcnn = RCNNBlock(in_channels, out_channels, t)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.rcnn(x) + self.conv_res(x)
