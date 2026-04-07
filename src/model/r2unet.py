import torch.nn as nn
import torch
from src.encoder.encoder import Encoder
from src.decoder.decoder import Decoder
from src.head.segmentation_head import SegmentationHead

class R2UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, t=2):
        super(R2UNet, self).__init__()
        self.encoder = Encoder(in_channels, t)
        self.decoder = Decoder(t)
        self.head = SegmentationHead(64, out_channels)

    def forward(self, x):
        x1, x2, x3, b = self.encoder(x)
        d1 = self.decoder(x1, x2, x3, b)
        out = self.head(d1)
        return out

if __name__ == "__main__":
    model = R2UNet()
    x = torch.randn(1,1,256,256)
    y = model(x)
    print(y.shape)  
