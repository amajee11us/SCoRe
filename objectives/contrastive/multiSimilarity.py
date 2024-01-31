from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from pytorch_metric_learning import losses

# for testing
from torch.autograd import Variable
import numpy as np


class MSLoss(nn.Module):
    def __init__(self, alpha=2, beta=50, margin=0.5):
        '''
        Parameters:
        alpha : Weight of the positive terms. Set to 2 in paper.
        beta : Weight of the negative terms. Set to 50 in paper.
        margin : Margin term for the objective function. default is 0.5. 
        '''
        super(MSLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.loss_func = losses.MultiSimilarityLoss(alpha=self.alpha, beta=self.beta, base=self.margin)
    
    def forward(self, features, labels):
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        repeat_count = features.shape[1]
        labels = labels.repeat(repeat_count)
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        return self.loss_func(features, labels)
