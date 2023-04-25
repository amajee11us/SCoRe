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
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GraphCut, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda
        self.epsilon = epsilon
        self.device = device

        self.temperature = temperature
        self.contrast_mode = 'all'
        self.base_temperature = 0.07

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # for the supervised case create a new negative mask
        mask_neg = 1.0 - mask
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        if self.sim_metric == 'rbf_kernel':
            anchor_dot_contrast = torch.cdist(anchor_feature, contrast_feature,2)**2
            anchor_dot_contrast = 1 - anchor_dot_contrast
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
        # Min the similarity between negative set
        log_prob = torch.log(
            (self.lamda * (exp_logits * mask)).sum(1) / (exp_logits * mask_neg).sum(1)
        )

        loss = - (self.temperature / self.base_temperature) * log_prob
        
        loss = loss.view(anchor_count, batch_size).mean()

        return loss