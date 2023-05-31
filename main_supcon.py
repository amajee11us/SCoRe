from __future__ import print_function

import os
import sys
import argparse
import time
import math

import wandb

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from dataloader.cubs2011 import CUBS
from dataloader.imagenet import ImageNet

from util import TwoCropTransform, AverageMeter, scale_255
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, load_model
from networks.resnet_big import SupConResNet
# import the loss functions
from objectives import *

# Imbalanced dataset creation
import numpy as np
import random

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--wandb', default=False, type = bool,
                        help = 'Boolean variable to indicate whether to use wandb for logging')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path', 'cubs', 'imagenet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['TripletLoss', 'SubmodTriplet', 'SupCon', 'SubmodSupCon', 'LiftedStructureLoss', 'opl', 'fl', 'gc', 'gc_rbf', 'LogDet'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # class imbalance parameters
    parser.add_argument('--use_imbalanced', action='store_true',
                        help='using Class Imbalanced dataset')
    parser.add_argument('--imbalance_ratio', type=int, default=10,
                        help='Imbalance ratio for imbalanced data sampling')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--constant', action='store_true',
                        help='using fixed LR')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--resume_from', type=str, default='',
                        help='Checkpoint path to resume from.')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    
    if opt.use_imbalanced:
        opt.model_name = '{}_imbalanced_{}_ratio'.format(opt.model_name, opt.imbalance_ratio)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # setup wandb
    if opt.wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        wandb.init(project=os.getenv("WANDB_PROJECT_NAME"), 
                   config=opt, 
                   name="stage1_" + opt.model_name,
                   entity=os.getenv("WANDB_USER_NAME"), 
                   settings=wandb.Settings(code_dir="."))
    
    if os.path.exists(opt.resume_from):
        print("Found a checkpoint to resume from !!")

    # setup pseudo global variables
    opt.total_train_steps = 1
    opt.total_val_steps = 1

    return opt


def create_class_imbal_indices(dataset, imbalance_ratio):
    # Get all training targets and count the number of class instances
    targets = np.array(dataset.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)
    
    # compute the imbalanced class counts
    mu = np.power(1 / imbalance_ratio, 1 / (nb_classes - 1))
    imbal_class_counts = [int(max(class_counts) * np.power(mu, i)) for i in range(nb_classes)]
    imbal_class_counts = list(imbal_class_counts)
    print(imbal_class_counts)

    # get balanced class indices
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    # sample for imbalanced distribution
    imbal_class_indices = [random.sample(list(class_idx), class_count) \
                           for class_idx, class_count in \
                            zip(class_indices, imbal_class_counts)]
    
    imbal_class_indices = np.hstack(imbal_class_indices)

    return imbal_class_indices

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'cubs':
        mean = (123., 117., 104.)
        std = (1., 1., 1.)
    elif opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = TwoCropTransform(train_transform)
    
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
    elif opt.dataset == 'imagenet':
        train_dataset = ImageNet(root=opt.data_folder, split='train',
                                 transform=train_transform)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=train_transform)
    elif opt.dataset == 'cubs':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=opt.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            scale_255(),
            normalize
        ])
        train_dataset = CUBS(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    if opt.use_imbalanced:
        imbal_class_idx = create_class_imbal_indices(train_dataset, opt.imbalance_ratio)
        train_dataset.data = train_dataset.data[imbal_class_idx]
        train_dataset.targets = np.array(train_dataset.targets)[imbal_class_idx]

        assert len(train_dataset.targets) == len(train_dataset.data)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    if opt.method == 'SupCon':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.method == 'SubmodSupCon':
        criterion = SubmodSupCon(temperature=opt.temp)
    elif opt.method == 'TripletLoss':
        criterion = TripletLoss(margin=0.5)
    elif opt.method == 'SubmodTriplet':
        criterion = SubmodTriplet(temperature=opt.temp)
    elif opt.method == 'fl':
        criterion = FacilityLocation(metric = 'cosSim', lamda = 1.0, use_singleton=False, temperature=opt.temp)
    elif opt.method == 'gc':
        criterion = GraphCut(metric = 'cosSim', lamda = 1.0, temperature=opt.temp, is_cf=True)
    elif opt.method == 'gc_rbf':
        criterion = GraphCut(metric = 'rbf_kernel', lamda = 1.0, temperature=opt.temp)
    # elif opt.method == 'LogDet':
    #     criterion = LogDet(metric = 'cosSim', temperature=opt.temp)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    running_loss = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        if opt.method in ['SupCon', 'SubmodSupCon', 'TripletLoss', 'SubmodTriplet', 'opl', 'fl', 'gc', 'gc_rbf', 'LogDet']:
            loss = criterion(features, labels)
        elif opt.method in ['LiftedStructureLoss']:
            #labels = torch.cat((labels,labels)) # TODO : Fix this later
            loss = criterion(features, labels)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.5f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        if opt.wandb:
            logger = {'Train_steps': opt.total_train_steps,
                    'Train_repr_loss': loss.item(),
                    'LR' : optimizer.param_groups[0]['lr'] 
                    }

            wandb.log(logger)   

        opt.total_train_steps += 1 
    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    if opt.wandb:
        wandb.watch(model)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    start_epoch = 1
    # Check if we have to resume and perform a resume operation
    if opt.resume_from != '':
        model, optimizer, start_epoch = load_model(opt.resume_from, 
                                                   model, optimizer)
        opt.total_train_steps = start_epoch * len(train_loader)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, loss {:.5f}, total time {:.2f}'.format(epoch, loss, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()