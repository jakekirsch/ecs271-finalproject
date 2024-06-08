import torch
import torch.nn as nn
from torchvision import models


class ModifiedDeepLabV3(nn.Module):
    def __init__(self, model):
        super(ModifiedDeepLabV3, self).__init__()
        self.backbone = model.backbone
        self.classifier = model.classifier
        self.upsample = nn.Upsample(size=(220, 220), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)  # across channel dim

    def forward(self, x):
        features = self.backbone(x)
        x = features['out']
        x = self.classifier(x)
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

def load_deeplab():
    import torch
    import torch.nn as nn
    from torchvision import models

    deeplab = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    deeplab.backbone.conv1 = nn.Conv2d(1, deeplab.backbone.conv1.out_channels,
                                       kernel_size=deeplab.backbone.conv1.kernel_size,
                                       stride=deeplab.backbone.conv1.stride,
                                       padding=deeplab.backbone.conv1.padding)
    # Modify the classifier to output 5 classes (or the desired number of segmentation classes)
    deeplab.classifier[4] = nn.Conv2d(in_channels=256, out_channels=5, kernel_size=(1, 1), stride=(1, 1))
    modified_model = ModifiedDeepLabV3(deeplab)
    # Print the modified model to verify changes
    print(modified_model)

    return modified_model