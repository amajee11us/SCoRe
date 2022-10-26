from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging

'''
Supervised Contrastive Loss: https://arxiv.org/pdf/2004.11362.pdf
'''

class SupervisedContrastiveLoss(nn.Module):
    '''
    Adapted from : https://github.com/HobbitLong/SupContrast
    '''
    def __init__(self, cfg, temperature=0.07, base_temperature=0.07, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        # Check the domain of the features
        if len(features.shape) < 2:
            raise ValueError("Only applicable to features as SupCon Loss is applied to samples.")
        
        elif len(features.shape) == 2:
            #raise ValueError("Only applicable to features as SupCon Loss is applied to samples.")
            features = torch.unsqueeze(features, dim=1)
        
        elif len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        #print(features.shape)

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            raise ValueError("Labels cannot be None in a supervised setting.")

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss