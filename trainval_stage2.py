from __future__ import print_function

import os
import sys
import argparse
import time
import math
import wandb
import numpy as np

# MedMNIST dataset
import medmnist
from medmnist import INFO # INFO allows us to fetch the details of dataset based on search flag

import torch
import torch.backends.cudnn as cudnn

from trainval_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, f1_loss, list_of_ints
from util import set_optimizer
from util import plot_tsne, generate_tsne_from_feat_embedding, plot_confusion_matrix
from networks.resnet_big import SupConResNet, LinearClassifier

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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--wandb', default=False, type = bool,
                        help = 'Boolean variable to indicate whether to use wandb for logging')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # class imbalance parameters
    parser.add_argument('--use_imbalanced', action='store_true',
                        help='using Class Imbalanced dataset')
    parser.add_argument('--imbalance_ratio', type=int, default=10,
                        help='Imbalance ratio for imbalanced data sampling')
    parser.add_argument('--imbalance_classes', type=list_of_ints, default=None,
                        help='List of class IDs in the range of [0, max_no_of_classes]')
    
    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 
                                 'cubs', 'imagenet', 
                                 'organamnist', 'dermamnist', 'bloodmnist'], help='dataset')

    # other setting
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SubmodSupCon', 
                                 'TripletLoss', 'SubmodTriplet', 
                                 'LiftedStructureLoss', 'NPairsLoss', 
                                 'MSLoss', 'SNNLoss', 'SubmodSNN',
                                 'fl', 'gc', 'gc_rbf', 'LogDet'], help='choose method')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--constant', action='store_true',
                        help='using fixed LR')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'organamnist':
        opt.n_cls = 11
    elif opt.dataset == 'dermamnist':
        opt.n_cls = 7
    elif opt.dataset == 'bloodmnist':
        opt.n_cls = 8
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'cubs':
        opt.n_cls = 200
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # setup wandb
    if opt.wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        wandb.init(project=os.getenv("WANDB_PROJECT_NAME"), 
                   config=opt, 
                   name="stage2_"+opt.model_name,
                   entity=os.getenv("WANDB_USER_NAME"), 
                   settings=wandb.Settings(code_dir="."))

    # setup pseudo global variables
    opt.total_train_steps = 1
    opt.total_val_steps = 1

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        labels = labels.view(-1)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
        
        if opt.wandb:
            logger = {'Train_steps': opt.total_train_steps,
                    'Train_clf_loss': loss.item(),
                    'Train_acc_1': acc1[0],
                    'Train_acc_5': acc5[0],
                    'Train_LR': optimizer.param_groups[0]['lr']
                    }

            wandb.log(logger) 

        opt.total_train_steps += 1    

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt, epoch):
    """validation"""
    model.eval()
    classifier.eval()

    # Store feature and label embeddings for processing t-SNE plots
    plot_features = []
    plot_labels = []
    plot_predictions = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            feat = model.encoder(images)
            output = classifier(feat)
            labels = labels.view(-1)
            loss = criterion(output, labels)

            plot_labels.append(labels.cpu().numpy())
            _, pred_labels = torch.max(output, 1)
            plot_predictions.append(pred_labels.cpu().numpy())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            f1 = f1_loss(output, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))     
    # Convert plot_labels and plot_predictions to NumPy arrays
    plot_labels_np = np.concatenate(plot_labels)
    plot_predictions_np = np.concatenate(plot_predictions) 
    
    if opt.wandb:
        logger = {'Val_steps': opt.total_val_steps,
                'Val_clf_loss': losses.avg,
                'Val_acc_1': top1.avg
                }

        wandb.log(logger) 
    
    opt.total_val_steps += 1

    print(' * Acc@1 {top1.avg:.3f} F1 {f1:.3f}'.format(top1=top1, f1=f1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)
    if opt.wandb:
        wandb.watch(model)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt, epoch)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
