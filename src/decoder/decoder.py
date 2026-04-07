import torch.nn as nn
import torch
from src.blocks.upsample_block import UpConv
from src.blocks.rrcnn_block import RRCNNBlock

class Decoder(nn.Module):
    def __init__(self, t=2):
        super(Decoder, self).__init__()
        self.up3 = UpConv(512, 256)
        self.d3 = RRCNNBlock(512, 256, t)
        self.up2 = UpConv(256, 128)
        self.d2 = RRCNNBlock(256, 128, t)
        self.up1 = UpConv(128, 64)
        self.d1 = RRCNNBlock(128, 64, t)

    def forward(self, x1, x2, x3, b):
        up3 = self.up3(b)
        d3 = self.d3(torch.cat([up3, x3], dim=1))
        up2 = self.up2(d3)
        d2 = self.d2(torch.cat([up2, x2], dim=1))
        up1 = self.up1(d2)
        d1 = self.d1(torch.cat([up1, x1], dim=1))
        return d1
