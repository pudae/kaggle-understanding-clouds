import torch
import torch.nn as nn
import torch.nn.functional as F

import kvt


def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)

    if input.dim() == 4:
        B,C,H,W = input.size()
        iflat = input.view(B*C,-1)
        tflat = target.view(B*C,-1)
    else:
        assert input.dim() == 3
        B,H,W = input.size()
        iflat = input.view(B,-1)
        tflat = target.view(B,-1)
    intersection = (iflat * tflat).sum(dim=1)
                
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth))
    loss = loss.mean()
    return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
        return dice_loss(input, target)
