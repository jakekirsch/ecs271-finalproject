import torch
import torch.nn as nn

class RectificationLoss(nn.Module):
    """
    Given input from two models and 1 label, implement the Contrastive Discrepancy Review loss
    """
    def __init__(self, classes = [0, 1, 2, 3, 4], weighted: bool = True):
        super().__init__()
        self.classes = classes
        self.weighted = weighted
        self.mse = weighted_MSELoss()
        
    def _probas_to_class_channels(self, probas_1):
        """
        Convert the probabilities of dim [batch x channel x height x width] to labels 0/1. 
        A 1 in the channel means that pixel is labeled that channel (e.g. [batch x 3 x h x w] = 1 --> that pixel is a label 3)
        """
        # Find the indices of the maximum values along the channels dimension
        max_indices = torch.argmax(probas_1, dim=1, keepdim=True)
        predicted_channels = torch.zeros_like(probas_1)
        predicted_channels = predicted_channels.scatter_(1, max_indices, 1)
        return predicted_channels
    
    def _targets_to_class_channels(self, targets):
        """
        Convert the targets [batch x h x w] to [batch x channel x h x w] where a 1 in the channel
        corresponds to a label
        """
        
        assert len(targets.shape) == 3, f"targets shape: {targets.shape} not 3"
        targets_shape = targets.shape
        bs = targets_shape[0]
        channels = len(self.classes)
        height = targets_shape[1]
        width = targets_shape[2]
        target_channels = torch.zeros((bs, channels, height, width))
        for c in self.classes:
            target_channels[:, c, :, :] = (targets == c).int()
        return target_channels
    
    def _disagreement_mask(self, predicted_channels_1, predicted_channels_2):
        return torch.logical_xor(predicted_channels_1, predicted_channels_2)
    
    
    
    def forward(self, probas_1, probas_2, targets):
        assert len(probas_1.shape) == 4, f"{probas_1.shape} is not correct dimension"
        assert len(probas_2.shape) == 4, f"{probas_2.shape} is not correct dimension"
        assert probas_1.shape == probas_2.shape, f"{probas_1.shape} != {probas_2.shape}"
        # per channel
        predicted_channels_1 = self._probas_to_class_channels(probas_1)
        predicted_channels_2 = self._probas_to_class_channels(probas_2)
        target_channels = self._targets_to_class_channels(targets)
        # get region of disagreement between the two predictions for each channel
        mask = self._disagreement_mask(predicted_channels_1, predicted_channels_2)
        # now clip the predictions to the masked area for each channel 
        clipped_pred_1 = torch.logical_and(mask, predicted_channels_1)
        clipped_pred_2 = torch.logical_and(mask, predicted_channels_2)
        clipped_y = torch.logical_and(mask, target_channels)
        
        
        # use size of the disagreement region as the weight
        weights = mask.sum(dim=(-1, -2))
        weights = weights.unsqueeze(-1).unsqueeze(-1) # add back the dimensions

        rectification_loss_1 = self.mse(clipped_pred_1.float(), clipped_y.float(), weights)
        rectification_loss_2 = self.mse(clipped_pred_2.float(), clipped_y.float(), weights)
        return rectification_loss_1, rectification_loss_2
        


class weighted_MSELoss(nn.Module):

    def forward(self, inputs, targets, weights):
        assert weights.shape[0] == inputs.shape[0], f"weights.shape: {weights.shape} not expected dimensions"
        assert weights.shape[1] == inputs.shape[1], f"weights.shape: {weights.shape} not expected dimensions"
        assert weights.shape[2] == 1, f"weights.shape: {weights.shape} not expected dimensions"
        assert weights.shape[3] == 1, f"weights.shape: {weights.shape} not expected dimensions"
        
        weighted_SE = (((inputs - targets)**2 ) * weights)
        # mean across H x W and Channel
        return weighted_SE.mean(dim=(-1, -2, -3))



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


    
    
    