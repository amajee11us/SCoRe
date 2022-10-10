from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
from lib.utils import get_target_device

class GraphCut(nn.Module):
    def __init__(self, metric='euclidean', lamda = 0.5):
        super(GraphCut, self).__init__()
        # determine the metric
        self.sim_metric = metric
        # determine the constant
        self.lamda = lamda

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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
      loss = 0
      X_set = X_set.reshape(X_set.shape[0], -1)

      for iter in range(unique_class_labels.shape[0]):
        pos_set, neg_set = [], []
        
        for fid in range(Y_set.shape[0]):
          if Y_set[fid] == unique_class_labels[iter]:
            pos_set.append(X_set[fid])
          else:
            neg_set.append(X_set[fid])
        
        # stack tensors 
        pos_set = torch.stack(pos_set)
        neg_set = torch.stack(neg_set)
        
        if self.sim_metric == 'euclidean':
          pos_dist_matrix = torch.cdist(pos_set, pos_set,2)
          neg_dist_matrix = torch.cdist(pos_set, neg_set,2)
        elif self.sim_metric == 'cosSim':
          pos_set_norm = torch.norm(pos_set, p=2, dim=1).unsqueeze(1).expand_as(pos_set)
          pos_set_normalized = pos_set.div(pos_set_norm + 1e-5)
          neg_set_norm = torch.norm(neg_set, p=2, dim=1).unsqueeze(1).expand_as(neg_set)
          neg_set_normalized = neg_set.div(neg_set_norm + 1e-5)
          
          pos_dist_matrix = torch.matmul(pos_set_normalized, pos_set_normalized.T)
          neg_dist_matrix = torch.matmul(pos_set_normalized, neg_set_normalized.T)

        pos_sum = torch.sum(pos_dist_matrix)
        neg_sum = torch.sum(neg_dist_matrix)


        if self.sim_metric == 'euclidean':
          loss += pos_sum - self.lamda * neg_sum
        else:
          loss += neg_sum - self.lamda * pos_sum

      return loss
    
    def __calculateSampleSeparation(self,x_i, x_j, exp = 2):

      n = x_i.size(0)
      m = x_j.size(0)
      d = x_i.size(1)

      x_i = x_i.unsqueeze(1).expand(n, m, d)
      x_j = x_j.unsqueeze(0).expand(n, m, d)

      dist = torch.pow(x_i - x_j, exp).sum(2) 
      return dist 