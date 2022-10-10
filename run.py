import torch
import logging
from tensorboardX import SummaryWriter
from torchsummary import summary

import os
import argparse
import pprint

# import module classes
from datasets import imagenet
# from dataset import mini_imagenet
from datasets import cifar10

# library imports
from lib.models import factory
from lib.solver import build_optimizer, build_lr_scheduler
from lib.dataset_factory import build_dataset
from lib.engine import train, validate, resume_from_ckpt
from lib.utils import *
from lib.config.conf import cfg_from_file
from lib.config.conf import __C as cfg

# combinatorial optimization
from lib.losses.graphCut import GraphCut
from lib.losses.supCon import SupervisedContrastiveLoss

# define logger
log = Logger(cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark for comparing Submodular Functions against Continuous Learning functions')

    # General parser
    parser.add_argument('-c',
                        '--config_file',
                        dest='config_file',
                        default='configs/alexnet_224x224.yaml',
                        help='model architecture (default: alexnet)')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Get configuration
    cfg_from_file(args.config_file)
    cfg.OUTPUT_DIR = get_output_ckpt_dir(cfg)

    log.info("Reading config from file: {}".format(args.config_file))

    log.info(pprint.PrettyPrinter(indent=4).pprint(cfg))
    # Select appropriate device
    device = get_target_device(cfg)
    log.info(f'Using {device} for execution.')
    '''
    Model/Optimizer setup
    '''
    tbwriter = SummaryWriter(log_dir=get_output_tb_dir(cfg))
    if args.seed is None:
        seed = torch.initial_seed()
    else:
        seed = torch.manual_seed(args.seed)
    log.info("Using Seed : {}".format(seed))

    # create model and load to device
    alexnet = factory.build_model(cfg)
    log.info(alexnet)

    #summary(alexnet, input_size=(3, 224, 224))

    # Create optimizer
    optimizer = build_optimizer(cfg, alexnet)
    # Create an LR scheduler, Multiply rate by 0.1 every LR_STEP
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Define the combinatorial objective
    #gc = GraphCut(metric = 'cosSim', lamda = 0.9)
    gc = SupervisedContrastiveLoss(cfg, temperature=0.1)
    #gc = None
    '''
    Resume from a checkpoint
    pass the model and the optimizer and load the stuff
    '''
    if not args.resume == None:
        resume_from_ckpt(args.resume, alexnet, optimizer)
    '''
    Load and prepare datasets
    Supported CIFAR10/Imagenet.
    '''
    train_loader = build_dataset(cfg, split="train")
    val_loader = build_dataset(cfg, split="val")
    '''
    Start Training with specified config.
    '''
    # Set the initial param for best accuracy to beat
    best_acc1 = 0

    # Train over the dataset
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # adjust_learning_rate(optimizer, epoch, cfg)

        # train one epoch on the target device
        train(train_loader, alexnet, criterion, optimizer, epoch, cfg, gc,
              writer=tbwriter)

        # Get the top1 accuracy from the validation set
        acc1 = validate(val_loader, alexnet, criterion, cfg, gc, writer=tbwriter)

        # step on the learning-rate
        lr_scheduler.step()

        check_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save_checkpoint(cfg, {
        #     'epoch': epoch + 1,
        #     'arch': cfg.ARCH,
        #     'state_dict': alexnet.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        #     'lr': get_lr(optimizer)
        # },
        #                 is_best=check_best)


if __name__ == "__main__":
    main()