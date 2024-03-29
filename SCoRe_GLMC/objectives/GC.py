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

    def forward(self, features, features_contrast, labels=None, label_weight=None):         
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))   
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        label_weight = label_weight.view(-1, 1) 
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_pos = label_weight * mask_pos        
        mask_neg = 1.0 - mask_pos
        
        # Remove the self similarity between samples
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)
        
        normalized_anchors = F.normalize(features, p=2, dim=1)
        normalized_contrast = F.normalize(features_contrast, p=2, dim=1)
        
        # Compute the Similarity Kernel
        sim_kernel = torch.mm(
                normalized_anchors, normalized_contrast.t())

        loss = torch.div(
            -self.lamda * sim_kernel.sum(1),
            mask.sum(1)
        )

        return loss.mean()