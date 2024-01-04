from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class LogDet(nn.Module):
    def __init__(self, metric='cosSim',
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       temperature=0.7,
                       is_cf=True):
        super(LogDet, self).__init__()
        
        self.device = device
        self.lamda = 0.1
        self.temperature = temperature
        #self.temperature = 1.0
        self.contrast_mode = 'all'
        self.base_temperature = 0.07
        self.is_cf = is_cf

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask_pos = torch.eq(labels, labels.T).float().to(device)
        else:
            mask_pos = mask_pos.float().to(device)

        # for the supervised case create a new negative mask
        mask_ground = torch.ones_like(mask_pos) 

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute similarity kernel
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        logits = anchor_dot_contrast

        # get unique vectors 
        labelSet = torch.unique(labels, sorted=True)
        labels = labels.view(-1).repeat(contrast_count)
        labels = labels.unsqueeze(1)
        mask_ground = mask_ground.repeat(anchor_count, contrast_count)

        loss = 0.0        
        for label_index in labelSet:
            mask_set = torch.where(labels == label_index, 1.0, 0.0)
            indices = torch.nonzero(mask_set.squeeze(1))
            S_label = torch.index_select(logits, 0, indices.squeeze(1))
            S_label = torch.index_select(S_label, 1, indices.squeeze(1))

            loss +=  torch.logdet(S_label + (0.5 * torch.eye(S_label.shape[0]).to(device))) 
        
        if self.is_cf:
            ground_set_det = torch.logdet((logits * mask_ground) + (0.5 * torch.eye(mask_ground.shape[0]).to(device)))
            loss -= ground_set_det
        else:
            loss = - (self.temperature / self.base_temperature) * (loss /labelSet.shape[0])

        return loss