import torch
import torch.nn as nn
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.features = original_model
        self.dense = nn.Conv2d(512, 5, kernel_size=1)  # Add a dense layer to produce 5 output channels
        self.upsample = nn.Sequential(
            nn.Conv2d(5, 5, kernel_size=3, padding=2),
            nn.Upsample(size=(220, 220), mode='bilinear', align_corners=False)
        )
        # for outputs for plotting
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) # across channel dim

    def forward(self, x):
        x = self.features(x)
        x = self.dense(x)
        x = self.upsample(x)
        return x
    
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
    
    
def load_resnet():
    import torch
    import torch.nn as nn
    from torchvision import models
    
    # Load a pretrained ResNet model
    resnet = models.resnet18(weights='DEFAULT')
    # Modify the input layer to accept 1 channel instead of 3
    resnet.conv1 = nn.Conv2d(1, resnet.conv1.out_channels, kernel_size=resnet.conv1.kernel_size,
                             stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=False)

    resnet = nn.Sequential(*list(resnet.children())[:-2])
    modified_resnet = ModifiedResNet(resnet)
    return modified_resnet

# model = load_resnet()
# # Create a sample input tensor with 1 channel
# sample_input = torch.randn(229, 1, 320, 320)  # Batch size = 1, Channels = 1, Height = 224, Width = 224

# # Set the model to evaluation mode
# model.eval()

# # Pass the input tensor through the model
# with torch.no_grad():
#     output = model(sample_input)

# # Print the output
# print(output.shape)
