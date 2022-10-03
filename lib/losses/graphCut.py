from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from lib.utils import get_target_device

class GraphCut(nn.Module):
    def __init__(self, num_classes, input_dims, batch_size, lamda = 0.5):
        super(GraphCut, self).__init__()
        self.num_input = batch_size
        self.input_dims = input_dims
        self.num_classes = num_classes

        # determine the constant
        self.lamda = lamda

    def forward(self, X_set, Y_set):
      '''
      X_set : this is the feature set from the feature extractor
      Y_set : This is a set of classes corresponding to each x in X_set
      
      1. select the X_set from class Y_set - positive class
      2. select all other X_set from Y_set not in step 1 - different classes
      3. calculate the separation between positive classes in batch and maximize it (euc distance)
      4. Calculate the separation between negative classes in  batch and minimize it (euc distance).
      '''
      unique_class_labels = torch.unique(Y_set)

      for cls in unique_class_labels:
        pos_set, neg_set = [], []

        for fid in range(Y_set.shape[0]):
          if Y_set[fid] == cls:
            pos_set.append(X_set[fid])
          else:
            neg_set.append(X_set[fid])
        
        pos_dist_matrix = torch.cdist(torch.tensor(pos_set), torch.tensor(pos_set),2)
        neg_dist_matrix = torch.cdist(torch.tensor(pos_set), torch.tensor(neg_set),2)
        pos_sum = torch.sum(pos_dist_matrix)
        neg_sum = torch.sum(neg_dist_matrix)

        return neg_sum - self.lamda * pos_sum
    
    def __calculateSampleSeparation(self,x_i, x_j, exp = 2):

      n = x_i.size(0)
      m = x_j.size(0)
      d = x_i.size(1)

      x_i = x_i.unsqueeze(1).expand(n, m, d)
      x_j = x_j.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x_i - x_j, exp).sum(2) 
      return dist 