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


class NPairsLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.5):
        '''
        Parameters:
        margin : Margin term for the objective function. default is 0.5. 
        '''
        super(NPairsLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.NTXentLoss(temperature=temperature)
        #self.loss_func = losses.NPairsLoss()
    
    def forward(self, features, labels):
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        repeat_count = features.shape[1]
        labels = labels.repeat(repeat_count)
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        return self.loss_func(features, labels)
