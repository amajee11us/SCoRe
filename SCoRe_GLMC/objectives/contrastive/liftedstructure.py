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


class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=1, beta=0, margin=0.5, hard_mining=True):
        '''
        Parameters:
        alpha : Negative margin for the LSL loss. In paper it is set to 1.
        beta : Positive margin for the LSL loss. In paper it is set to 0.
        margin : Margin term for hard sample mining task. default is 0.5. 
        hard_mining : Whether to perform hard-sample mining. By default the loss mines only negative samples.
        '''
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining
        self.loss_func = losses.LiftedStructureLoss(neg_margin=self.alpha, pos_margin=self.beta)
    
    def forward(self, features, labels):
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        repeat_count = features.shape[1]
        labels = labels.repeat(repeat_count)
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        return self.loss_func(features, labels)