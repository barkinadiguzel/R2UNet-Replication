import torch.nn as nn
import torch
from src.blocks.rrcnn_block import RRCNNBlock

class Encoder(nn.Module):
    def __init__(self, in_channels, t=2):
        super(Encoder, self).__init__()
        self.e1 = RRCNNBlock(in_channels, 64, t)
        self.e2 = RRCNNBlock(64, 128, t)
        self.e3 = RRCNNBlock(128, 256, t)
        self.bottleneck = RRCNNBlock(256, 512, t)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        b = self.bottleneck(self.pool(x3))
        return x1, x2, x3, b
