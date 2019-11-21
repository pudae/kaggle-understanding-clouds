import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricBCELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        y_true = target
        y_pred = torch.sigmoid(input)

        y_true = y_true.view(-1, 1)
        y_pred = y_pred.view(-1, 1)

        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)

        loss_ce = F.kl_div(torch.log(y_pred_1), y_true_1)
        loss_rce = F.kl_div(torch.log(y_true_2), y_pred_2)
        return self.alpha*loss_ce + self.beta*loss_rce
