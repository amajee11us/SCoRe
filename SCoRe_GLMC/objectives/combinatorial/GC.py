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

    def forward(self, features, features_contrast):
        batch_size = features.shape[0]
        
        normalized_anchors = F.normalize(features, p=2, dim=1)
        normalized_contrast = F.normalize(features_contrast, p=2, dim=1)
        
        # Compute the Similarity Kernel
        sim_kernel = torch.mm(
                normalized_anchors, normalized_contrast.t())

        loss = torch.div(
            -self.lamda * sim_kernel.sum(1),
            mask.sum(1)
        )
        
        # loss = - (self.temperature / self.base_temperature) * loss
        
        # loss = loss.view(anchor_count, batch_size).mean()

        return loss.mean()