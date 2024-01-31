from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from pytorch_metric_learning import losses

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.temperature = 0.7
        self.loss_func = losses.TripletMarginLoss(margin=self.margin,
                                                swap=False,
                                                smooth_loss=False,
                                                triplets_per_anchor="all")

    def forward(self, features, labels=None):
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        repeat_count = features.shape[1]
        labels = labels.repeat(repeat_count)
        
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        return self.loss_func(features, labels)