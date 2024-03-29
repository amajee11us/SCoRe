import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5, temp=0.7):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.temperature = temp

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        tile = features.shape[1] # no of repetitions to be made
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        mask = mask.repeat(tile, tile)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.div(torch.matmul(features, features.t()), self.temperature)

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss