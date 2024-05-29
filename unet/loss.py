import torch
import torch.nn as nn


class GeneralizedDiceLoss(nn.Module):
    """
    https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, classes = torch.tensor([1, 2, 3, 4]), epsilon=1e-6, ):
        super(GeneralizedDiceLoss, self).__init__()
        self.classes = classes 
        # normalize the logits, since each channel is predicting its own class
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) # channel

        self.epsilon = epsilon

    def normalization(self, input):
        ## sigmoid to squash inputs
        ## softmax across channels to get to probabilities
        logits = self.sigmoid(input)
        proba = self.softmax(logits)
        return proba


    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        # compute per batch item multi-label dice
        per_item_dice = self.dice(input, target)
        batch_losses = 1. - per_item_dice
        # sum across batches
        return batch_losses.sum()
    
    def dice_per_channel(self, input, target, weighted=True):
        """
        input shape expects batch x class x h x w
        target shape expects batch x h x w
        """
        # need to transform target into a multi-channel tensor to align with preds
        multi_channel_target = torch.stack([(target == c).int() for c in self.classes], dim=1)
        # Sum the boolean masks to get the count of each class
        if weighted:
            target_class_weights = torch.sum(multi_channel_target, dim=(-1, -2))
            class_weights = 1 / (target_class_weights * target_class_weights).clamp(1e-6)
            class_weights.requires_grad = False
            intersection = (input * multi_channel_target).sum(dim=(-1, -2))
            union = (input + multi_channel_target).sum(dim=(-1, -2))
            return (2*intersection*class_weights).clamp(1e-6) / (union*class_weights).clamp(1e-6) # smoothing
        else:
            intersection = (input * multi_channel_target).sum(dim=(-1, -2))
            union = (input + multi_channel_target).sum(dim=(-1, -2))
            return (2*intersection).clamp(1e-6) / (union).clamp(1e-6) # smoothing
            

    def dice(self, input, target):
        """
        input shape expects batch x class x h x w
        target shape expects batch x h x w
        """
        dice_per_channel = self.dice_per_channel(input, target, weighted=True)
        avg_dice = dice_per_channel.mean(dim=1)
        return avg_dice

    
def print_dice_by_category(category_means, labels = {0: 'background', 1: 'esophagus', 2: 'heart', 3: 'trachea', 4: 'aorta'}):
    output = []    
    for i, value in enumerate(category_means):
        label = labels.get(i, f"Unknown label {i}")
        output.append(f"{label}: {value:.4f}")

    # Join all label-value pairs into a single string with comma separation
    result = ', '.join(output)
    print(result)

