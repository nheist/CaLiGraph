import torch
from torch import nn
import entity_linking.util as el_util


# loss strategies used for training
LOSS_MSE = 'MSE'
LOSS_NPAIR = 'NPAIR'
LOSS_NPAIRMSE = 'NPAIR+MSE'

LOSS_TYPES = [LOSS_MSE, LOSS_NPAIR, LOSS_NPAIRMSE]



class NpairLoss(nn.Module):
    """
    Computes multi-class N-pair loss according to
    http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
    but assumes that the labels of all entities in input are different.

    Adapted from https://github.com/ChaofWang/Npair_loss_pytorch/blob/master/Npair_loss.py
    """

    def __init__(self, l2_reg=0.02):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.l2_reg = l2_reg

    def forward(self, input, target):
        batch_size = input.size(0)
        labels = torch.arange(batch_size).to(el_util.DEVICE)
        logit = torch.matmul(input, torch.transpose(target, 0, 1))
        loss_ce = self.cross_entropy(logit, labels)

        l2_loss = torch.sum(input ** 2) / batch_size + torch.sum(target ** 2) / batch_size

        return loss_ce + self.l2_reg * l2_loss * 0.25


class NpairMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.npair_loss = NpairLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.npair_loss(input, target) + self.mse_loss(input, target)

