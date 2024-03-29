from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class SubmodSNN(nn.Module):
    def __init__(self, temperature=0.7,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(SubmodSNN, self).__init__()
        # determine the constant
        self.device = device

        self.temperature = temperature
        self.base_temperature = 0.07

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]    
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # for the supervised case create a new negative mask
        mask = torch.eq(labels, labels.T).float().to(device)
        mask_neg = 1.0 - mask
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_neg = mask_neg.repeat(anchor_count, contrast_count)

        # zero diagonal
        mask.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)
        
        dist_mat = self.pairwise_euclid_distance(anchor_feature, contrast_feature)
        sim_mat = self.pairwise_cos_distance(anchor_feature, contrast_feature)
        # compute logits
        sim_mat = torch.div(sim_mat, self.temperature)
        dist_mat = torch.div(dist_mat, self.temperature)

        # term one - distance term log sum exp
        logits = torch.log(
            torch.exp(dist_mat * mask).sum(1)
        )

        logits += torch.log(
            torch.exp(sim_mat * mask_neg).sum(1)
        ) 

        loss  = (self.temperature/self.base_temperature) * logits
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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
        return prod