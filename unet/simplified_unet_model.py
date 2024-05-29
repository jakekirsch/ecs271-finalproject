import torch
from torch import nn 
from unet.unet_layers import DoubleConv, Down, Up


class SimplifiedUNet(nn.Module):
    """
    Implement UNet described in 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/pdf/1505.04597v1
    """
    def __init__(self, n_channels: int, n_classes: int = 2, dropout: float = 0.2) -> None:
        """
        Default to binary classification for our project
        """
        super(SimplifiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout = dropout
        self.inc = DoubleConv(n_channels, 64, dropout=self.dropout)
        self.down1 = Down(in_channels=64, out_channels=128, dropout=self.dropout)
        self.down2 = Down(in_channels=128, out_channels=256, dropout=self.dropout)
        self.down3 = Down(in_channels=256, out_channels=512, dropout=self.dropout)
        # self.down4 = Down(in_channels=512, out_channels=1024)
        # bottom of U
        # self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256, dropout=self.dropout)
        self.up3 = Up(in_channels=256, out_channels=128, dropout=self.dropout)
        self.up4 = Up(in_channels=128, out_channels=64, dropout=self.dropout)
        # finally map each of the 64 feature maps to n_classes with fully connected layer
        self.out = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

        # for outputs for plotting
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) # across channel dim
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on input X, segThor data is 1x512x512 with
        multiple images/slices per patient
        """
        # print(f"Input x shape: {x.shape}")
        x1 = self.inc(x)
        # print(f"x1 shape: {x1.shape}")
        x2 = self.down1(x1)
        # print(f"x2 shape: {x2.shape}")
        x3 = self.down2(x2)
        #print(f"x3 shape: {x3.shape}")
        x4 = self.down3(x3)
        # print(f"x4 shape: {x4.shape}")
        # x5 = self.down4(x4)
        # bottom of U is 1024 feature maps of 24x24
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        #print(f"after up2 shape: {x.shape}")
        x = self.up3(x, x2)
        #print(f"after up3 shape: {x.shape}")
        x = self.up4(x, x1)
        #print(f"after up4 shape: {x.shape}")
        x = self.out(x)
        #print(f"after out shape: {x.shape}")
        logits = x # we expect this to be 2 channels of output pixels

        return logits


    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an x input, do the forward pass 
        but then 
        1. squash outputs using sigmoid
        2. get predicted probabilities by taking softmax
        
        Returns B x C x H x W where sum over the Channel dimension = 1. Each value is predicted probability of 
        the class C.
        """
        outputs = self.forward(x)
        logits = self.sigmoid(outputs)
        probas = self.softmax(logits)
        return probas

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an x input, get predicted probabilities B x C x H x W and 
        collapse to B x H x W where the value in each pixel is [0: C] based 
        on max probability across the input channels
        """
        probas = self.predict_probabilities(x)
        max_values, max_indices = torch.max(probas, dim=1)
        return max_indices
    
    def predict_class_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an x input, get predicted classes B x C x H x W where 
        the channel dimension corresponds to the label
        """
        probas = self.predict_probabilities(x)
        # Find the indices of the maximum values along the channels dimension
        max_indices = torch.argmax(probas, dim=1, keepdim=True)
        predicted_channels = torch.zeros_like(probas)
        predicted_channels = predicted_channels.scatter_(1, max_indices, 1)
        return predicted_channels