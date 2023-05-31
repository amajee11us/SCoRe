from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class GraphCut(nn.Module):
    def __init__(self, metric='euclidean', 
                       lamda = 0.5, 
                       epsilon = 10.0,
                       temperature=0.7,
                       is_cf=False,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GraphCut, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda
        self.epsilon = epsilon
        self.device = device
        self.is_cf = is_cf
        self.temperature = temperature
        self.contrast_mode = 'all'
        self.base_temperature = 0.07

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        # for the supervised case create a new negative mask
        mask_neg = 1.0 - mask
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        if self.sim_metric == 'rbf_kernel':
            anchor_feature = anchor_feature[:, None, :] # shape (N, D) -> (N, 1, D)
            contrast_feature = contrast_feature[None, :, :] # shape (N, D) -> (1, N, D)
            anchor_dot_contrast = torch.sum((anchor_feature - contrast_feature)**2, 2)
            anchor_dot_contrast = torch.exp(-anchor_dot_contrast/(0.1*anchor_dot_contrast.mean()))
        elif self.sim_metric == 'cosSim':
            anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        
        # compute logits
        anchor_dot_contrast = torch.div(
            anchor_dot_contrast,
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_neg = mask_neg.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask_neg.fill_diagonal_(0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # # V3
        log_prob = torch.log(exp_logits.sum(1, keepdim=True))
        if self.is_cf:
            log_prob = torch.div(
                -self.lamda * (exp_logits * mask_neg).sum(1),
                mask.sum(1)
            )
        else:
            # Min the similarity between negative set
            log_prob = torch.log(
                (self.lamda * (exp_logits * mask)).sum(1) / (exp_logits * mask_neg).sum(1)
            )

        loss = - (self.temperature / self.base_temperature) * log_prob
        
        loss = loss.view(anchor_count, batch_size).mean()

        return loss