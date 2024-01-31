import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PaCoFLLoss(nn.Module):
    def __init__(self, alpha, 
                       beta=1.0, gamma=1.0, supt=1.0, 
                       temperature=1.0, base_temperature=None, 
                       K=8192, 
                       num_classes=1000, 
                       smooth=0.5):
        super(PaCoFLLoss, self).__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        self.weight = None
        self.lamda = 1.0
        self.effective_num_beta = 0.999

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        mask_neg = 1.0 - mask

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        mask_neg = mask_neg * logits_mask

        # compute Facility Location log probs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = torch.log(
            self.lamda * (exp_logits * mask_neg).sum(1, keepdim=True)
        )

        # loss
        rew = self.class_weight.squeeze()[labels[:batch_size].squeeze()]
        loss = (self.temperature / self.base_temperature) * log_prob * rew
        loss = loss.mean()

        loss_balancesoftmax = F.cross_entropy(sup_logits+self.weight, labels[:batch_size].squeeze())
        return loss_balancesoftmax + self.alpha * loss
