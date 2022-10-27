import torch
import logging
from tensorboardX import SummaryWriter
from torchsummary import summary
import wandb
import os
from dotenv import load_dotenv
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
from lib.losses.facilityLocation import FacilityLocation

# define logger
log = Logger(cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark for comparing Submodular Functions against Continuous Learning functions')

    # General parser
    parser.add_argument('-c',
                        '--config_file',
                        dest='config_file',
                        default='configs/cifar_10_32x32.yaml',
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
    parser.add_argument('--rep_loss',
                        default='gc',
                        type=str,
                        help='Representation learning loss objective')
    parser.add_argument('--sim_metric',
                        default='rbf_kernel',
                        type=str,
                        help='Similarity metric for combinatorial representation learning loss objectives')
    parser.add_argument('--wandb',
                        default=False,
                        type = bool,
                        help = 'Boolean variable to indicate whether to use wandb for logging')
    parser.add_argument('--gpu_num',
                        default=1,
                        type = int,
                        help = 'GPU device number to use for training')
    args = parser.parse_args()
    return args


def main():
    # load environment variables from .env file
    load_dotenv()
    # load arguments from argparser
    args = parse_args()
    # Get configuration
    cfg_from_file(args.config_file)
    cfg.OUTPUT_DIR = get_output_ckpt_dir(cfg)
    cfg.GPU = [args.gpu_num]

    log.info("Reading config from file: {}".format(args.config_file))
    log.info(pprint.PrettyPrinter(indent=4).pprint(cfg))
    # Select appropriate device
    device = get_target_device(cfg, args.gpu_num)
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

    if args.wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        wandb.init(project=os.getenv("WANDB_PROJECT_NAME"), config=cfg, entity=os.getenv("WANDB_USER_NAME"), settings=wandb.Settings(code_dir="."))
    
    # create model and load to device
    model = factory.build_model(cfg, args.gpu_num)
    log.info(model)
    if args.wandb:
        wandb.watch(model)

    #summary(alexnet, input_size=(3, 224, 224))

    # Create optimizer
    optimizer = build_optimizer(cfg, model)
    # Create an LR scheduler, Multiply rate by 0.1 every LR_STEP
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Define the combinatorial objective
    if args.wandb:
        wandb.config.repr_loss = args.rep_loss
        wandb.config.sim_metric = args.sim_metric
        
    if args.rep_loss == 'fl':
       gc = FacilityLocation(metric = args.sim_metric, lamda = 0.9, use_singleton=False, device=device)
    elif args.rep_loss == 'gc':
       gc = GraphCut(metric = args.sim_metric, lamda = 0.9, device=device)
    elif args.rep_loss == 'supcon':
        gc = SupervisedContrastiveLoss(cfg, temperature=0.1, device=device)
    else:
        gc = None
    #gc = None
    '''
    Resume from a checkpoint
    pass the model and the optimizer and load the stuff
    '''
    if not args.resume == None:
        resume_from_ckpt(args.resume, model, optimizer)
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

    acc1 = validate(val_loader, model, criterion, -1, cfg, comb_optim=gc, writer=tbwriter, wandb_var=args.wandb, device=device)

    # Train over the dataset
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        # adjust_learning_rate(optimizer, epoch, cfg)

        # train one epoch on the target device
        train(train_loader, model, criterion, optimizer, epoch, cfg, comb_optim=gc,
              writer=tbwriter, wandb_var = args.wandb, device=device)

        # Get the top1 accuracy from the validation set
        acc1 = validate(val_loader, model, criterion, epoch, cfg, comb_optim=gc, writer=tbwriter, wandb_var = args.wandb, device=device)

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