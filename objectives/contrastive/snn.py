from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class SNNLoss(nn.Module):
    def __init__(self,
               temperature=100.,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        super(SNNLoss, self).__init__()
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
        self.STABILITY_EPS = 0.00001
    
    def fits(self, A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = self.pairwise_cos_distance(A, B)
        else:
            distance_matrix = self.pairwise_euclid_distance(A, B)
            
        return torch.exp(-(distance_matrix / temp))

    def pick_probability(self, x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        if len(x.shape) > 3:
            x = x.view(x.shape[0], x.shape[1], -1)
        if len(x.shape) == 3:
            x = torch.cat(torch.unbind(x, dim=1), dim=0)

        f = self.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0]).to(self.device)
        return f / (self.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))
    
    def same_label_mask(self, y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        y = y.contiguous().view(-1, 1)
        y2 = y2.contiguous().view(-1, 1)
        mask = torch.eq(y, y2.T).float().to(self.device)
        return mask
    
    def masked_pick_probability(self, x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        if len(x.shape) >= 3:
            mask = self.same_label_mask(y, y).repeat(x.shape[1], x.shape[1])
        else:
            mask = self.same_label_mask(y, y)
        
        mask.fill_diagonal_(0)

        return self.pick_probability(x, temp, cos_distance) * mask

    def forward(self, features, labels=None):
        self.device = (torch.device('cuda')
                       if features.is_cuda
                       else torch.device('cpu'))

        summed_masked_pick_prob = self.masked_pick_probability(
                                                        features, labels, 
                                                        self.temperature, 
                                                        self.cos_distance).sum(axis=1)
        return -torch.log(self.STABILITY_EPS + summed_masked_pick_prob).mean()
    
    def pairwise_euclid_distance(self, A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    def pairwise_cos_distance(self, A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        prod = torch.matmul(A, B.T)
        return 1 - prod
