from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from easydict import EasyDict as edict

__C = edict()

cfg = __C
'''
General
'''
__C.NUM_WORKERS = 2

__C.GPU = [0]  # pass a set of GPUS for data-parallel

__C.PRINT_FREQUENCY = 10

__C.ARCH = "alexnet"

__C.EXP_NAME = "basic_224"

__C.DEVICE = "cpu"

__C.OUTPUT_DIR = "output"
'''
Training setting
'''
__C.TRAIN = edict()

__C.TRAIN.LEARNING_RATE = 0.01

__C.TRAIN.NUM_EPOCHS = 90

__C.TRAIN.GAMMA = 0.1

__C.TRAIN.BATCH_SIZE = 12

__C.TRAIN.IMAGE_SIZE = 224

__C.TRAIN.NUM_CLASSES = 1000

__C.TRAIN.DATASET = edict()
__C.TRAIN.DATASET.NAME = "imagenet"
__C.TRAIN.DATASET.RESIZE = False
__C.TRAIN.DATASET.RANDOM_CROP = False
__C.TRAIN.DATASET.CROP_SIZE = 224
__C.TRAIN.DATASET.RANDOM_FLIP = False
__C.TRAIN.DATASET.NORM_MEAN = [0.485, 0.456, 0.406]
__C.TRAIN.DATASET.NORM_STD_DEV = [0.229, 0.224, 0.225]
__C.TRAIN.DATASET.PIL = False
__C.TRAIN.DATASET.CENTER_CROP = False
__C.TRAIN.DATASET.CENTER_CROP_SIZE = 224  # this can be used as a crop within a crop

__C.TRAIN.DATA_DIR = 'data/imagenet'

__C.TRAIN.DROPOUT_RATE = 0.5

__C.TRAIN.OPTIMIZER = "adam"

__C.TRAIN.SOLVER_LR = 0.0001  # Low learning rate to eradicate vanishing/ exploding gradients

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.NESTEROV = False

__C.TRAIN.LR_SCHEDULER = "steplr"

__C.TRAIN.LR_DECAY_STEP = 30  # for Step LR function

__C.TRAIN.LR_DECAY_MULTI_STEP = [20,
                                 40]  # for multi step LR scheduler function
'''
Testing Settings
'''
__C.TEST = edict()

__C.TEST.IMAGE_SIZE = 224

__C.TEST.DATASET = edict()
__C.TEST.DATASET.NAME = "imagenet"
__C.TEST.DATASET.RESIZE = False
__C.TEST.DATASET.RANDOM_CROP = False
__C.TEST.DATASET.CROP_SIZE = 224
__C.TEST.DATASET.RANDOM_FLIP = False
__C.TEST.DATASET.NORM_MEAN = [0.485, 0.456, 0.406]
__C.TEST.DATASET.NORM_STD_DEV = [0.229, 0.224, 0.225]
__C.TEST.DATASET.PIL = False
__C.TEST.DATASET.CENTER_CROP = False
__C.TEST.DATASET.CENTER_CROP_SIZE = 224  # this can be used as a crop within a crop

__C.TEST.DATA_DIR = 'data/imagenet'
__C.TEST.BATCH_SIZE = 12