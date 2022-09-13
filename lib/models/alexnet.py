from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

# data libraries
import torchvision
from torchvision import transforms

class AlexNet(nn.Module):

    def __init__(self, cfg=None):
        super(AlexNet, self).__init__() # initialize the nn.Module block

        self.conv_feat = nn.Sequential(
            # block 1 CONV --> ReLU --> POOl
            nn.Conv2d(3, 96, kernel_size= 11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2 CONV --> ReLU --> POOL
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3 CONV --> RelU
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # block 4 CONV --> ReLU
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # block 5 CONV --> ReLU --> POOL
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.TRAIN.DROPOUT_RATE), # dropouts cannot be in-place methods
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=cfg.TRAIN.DROPOUT_RATE),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, cfg.TRAIN.NUM_CLASSES),
        )

        self.__init_weights()

    def forward(self, x):
        '''
        Forward pass:
         Executes a single forward pass over the network on a batched input of sze b
        '''
        x = self.conv_feat(x)
        #Now average pool and flatten : increases precision
        x = self.avgpool(x)
        x = x.reshape(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input

        x = self.classifier(x)
        # return an array of (NUM_CLASSES dim with probabilities of each class) x batch_size
        return x

    def __init_weights(self):
        '''
        Weight initialization block:
         Weights are random initialized in a range
         biases are initialized to constant tensor 0
        '''

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
