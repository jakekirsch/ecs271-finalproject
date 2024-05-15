import torch
from torch import nn 
from typing import Optional
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Convolution layer, followed by BatchNorm then ReLU. Twice
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # padding = 0 and kernel_size means the we will reduce the HxW down, if 512-->510
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling layer with MaxPool2D stride 2, followed by DoubleConv layer
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_sample(x)
    
class Up(nn.Module):
    """
    Upsampling layer using nn.ConvTranspose2D
    """
    def __init__(self, in_channels, out_channels, dropout: float):
        super().__init__()
        # this is the "inverse" of Conv2D
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1) # upscale using the ConvTranspose2d
        # assuming input is channel x height x width we need to calculate the 
        # crop x2 down to the same size for concatenate
        dH = x2.size()[2] - x1.size()[2] # difference in the height
        dW = x2.size()[3] - x1.size()[3] # difference in the width
        left = dW // 2 
        right = dW - dW // 2
        top = dH // 2
        bottom = dH - dH // 2

        x2 = x2[:,:,bottom:-top,left:-right]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)