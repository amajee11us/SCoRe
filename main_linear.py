from __future__ import print_function

import os
import sys
import argparse
import time
import math
import wandb

import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, f1_loss
from util import set_optimizer
from util import plot_tsne, generate_tsne_from_feat_embedding
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
    parser.add_argument('--comet', action='store_true',
                        help="Boolean argument for comet logging")
    

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
    
    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cubs', 'imagenet', 'imagenet32'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'cubs':
        opt.n_cls = 200
    elif opt.dataset in ['imagenet', 'imagenet32']:
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


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, experiment=None):
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

        if opt.comet:
            logger = {'Train_steps': opt.total_train_steps,
                    'Train_clf_loss': loss.item(),
                    'Train_acc_1': acc1[0],
                    'Train_acc_5': acc5[0],
                    'Train_LR': optimizer.param_groups[0]['lr']
                    }

            experiment.log_metrics(logger, step=opt.total_train_steps)
            
        opt.total_train_steps += 1


    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt, epoch, experiment=None):
    """validation"""
    model.eval()
    classifier.eval()

    # Store feature and label embeddings for processing t-SNE plots
    plot_features = []
    plot_labels = []

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
            loss = criterion(output, labels)

            # update the features in the t-SNE lists
            plot_features.append(feat)
            plot_labels.append(labels)

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

    embeddings, labels = generate_tsne_from_feat_embedding(plot_features, plot_labels)
    plot_tsne(embeddings, labels, 10, epoch+1, "stage2_"+opt.model_name)       
    
    if opt.wandb:
        logger = {'Val_steps': opt.total_val_steps,
                'Val_clf_loss': losses.avg,
                'Val_acc_1': top1.avg
                }

        wandb.log(logger) 

    if opt.comet:
        experiment.log_metric("Val_steps", opt.total_val_steps, step=opt.total_val_steps)
        experiment.log_metric("Val_clf_loss", losses.avg, step=opt.total_val_steps)
        experiment.log_metric("Val_acc_1", top1.avg, step=opt.total_val_steps)
    
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

    if opt.comet:
        from comet_ml import Experiment
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="r96agak8trPzpcPhI9chgJo7F",
            project_name="score",
            workspace="krishnatejakk",
        )
    else:
        experiment = None

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt, experiment=experiment)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt, epoch, experiment=experiment)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
