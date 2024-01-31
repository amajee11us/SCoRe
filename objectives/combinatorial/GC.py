from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from objectives.utils import similarity_kernel

class GraphCut(nn.Module):
    def __init__(self, metric='euclidean', 
                       lamda = 0.5, 
                       temperature=0.7,
                       is_cf=False,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GraphCut, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda
        self.device = device
        self.is_cf = is_cf
        self.temperature = temperature
        self.base_temperature = 0.07

    def forward(self, features, labels=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        # Compute the label mask
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(self.device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_neg = 1.0 - mask
        mask.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        # Compute the Similarity Kernel
        sim_kernel = similarity_kernel(anchor_feature, 
                                       metric=self.sim_metric) 

        if self.is_cf:
            loss = torch.div(
                -self.lamda * (sim_kernel * mask_neg).sum(1),
                mask.sum(1)
            )
        else:
            # Min the similarity between negative set
            loss = torch.log(
                (self.lamda * (sim_kernel * mask)).sum(1) / (sim_kernel * mask_neg).sum(1)
            )

        loss = - (self.temperature / self.base_temperature) * loss
        
        loss = loss.view(anchor_count, batch_size).mean()

        return loss