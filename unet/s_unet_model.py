import torch
from torch import nn 
from unet.unet_layers import DoubleConv, Down, Up


class UNet(nn.Module):
    """
    Implement UNet described in 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/pdf/1505.04597v1

    # TODO: simplified UNet SegThor paper
    """
    def __init__(self, n_channels: int, n_classes: int = 2) -> None:
        """
        Default to binary classification for our project
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        # self.down4 = Down(in_channels=512, out_channels=1024)
        # bottom of U
        # self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)
        # finally map each of the 64 feature maps to n_classes with fully connected layer
        self.out = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on input X, segThor data is 1x512x512 with
        multiple images/slices per patient
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # bottom of U is 1024 feature maps of 24x24
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        logits = x # we expect this to be 2 channels of output pixels

        return logits
