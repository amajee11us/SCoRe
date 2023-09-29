from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging

class SubmodTriplet(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SubmodTriplet, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # Check the domain of the features
        if len(features.shape) < 3:
            raise ValueError("Only applicable to features as SupCon Loss is applied to samples.")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask_pos = torch.eq(labels, labels.T).float().to(self.device)
        else:
            raise ValueError("Labels cannot be None in a supervised setting.")
        mask_neg = 1.0 - mask_pos

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute Similarity Kernel /Matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T)**2,
            self.temperature)
        # for numerical stability - borrowed from SupCon paper
        max_sim_per_anchor, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        sim_kernel = anchor_dot_contrast - max_sim_per_anchor.detach()
        sim_kernel = sim_kernel 

        # create the positive and the negative masks
        mask_pos = mask_pos.repeat(anchor_count, contrast_count)
        mask_neg = mask_neg.repeat(anchor_count, contrast_count)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        # Compute the loss
        # compute the second term - sum_pos(log sum_neg(exp(S)))
        loss = (sim_kernel * mask_neg).sum(1) - (sim_kernel * mask_pos).sum(1)
        
        # compute mean of log-likelihood over the entire set
        mean_loss = loss #/ loss.shape[0]
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_loss
        loss = loss.view(anchor_count, batch_size).mean()

        return loss