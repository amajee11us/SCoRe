from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.temperature = 0.7

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # Check the domain of the features
        if len(features.shape) < 3:
            raise ValueError("Only applicable to features as Triplet Loss is applied to samples.")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        n = features.shape[0]
        tile = features.shape[1]

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        sim_mat = torch.div(
                        torch.matmul(features, features.T), 
                        self.temperature)
        labels = labels.contiguous().view(-1, 1)
        #labels = torch.cat((labels, labels))
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask.repeat(tile, tile)
        neg_mask = 1.0 - pos_mask
        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)
        
        hardest_pos_per_anchor,_ = torch.max(pos_mask * sim_mat, dim=1)
        hardest_neg_per_anchor,_ = torch.max(neg_mask * sim_mat, dim=1)

        per_sample_loss = torch.relu(
            hardest_pos_per_anchor - hardest_neg_per_anchor + self.margin)
        
        loss = per_sample_loss.mean()

        return loss