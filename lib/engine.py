import torch
import time
import logging
import wandb
from .meters import AverageMeter, ProgressMeter, accuracy, f1_loss
from .utils import get_lr
from lib.losses.graphCut import GraphCut

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
Keep track of the steps covered and write to TB
'''
total_train_steps = 1
total_val_steps = 1


def train(train_loader, model, criterion, optimizer, epoch, cfg, comb_optim=None, writer=None, wandb_var=False):
    '''
    Train over one epoch on a mini-batch
    '''
    # step count
    step_count = int(len(train_loader)/ cfg.TRAIN.BATCH_SIZE)
    #prev_steps = step_count * epoch
    global total_train_steps
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    f1 = AverageMeter('f1', ':6.2f')
    progress = ProgressMeter(step_count,
                             batch_time,
                             data_time,
                             losses,
                             top1,
                             top5,
                             f1,
                             prefix="Epoch: [{}]".format(epoch))

    #switch to training
    model.train()

    end = time.time()

    for i in range(step_count):
        images, label = next(iter(train_loader))
        data_time.update(time.time() - end)

        images = images[0].to(device)
        label = torch.flatten(torch.stack(label)).to(device)

        features, output = model(images)
        loss = criterion(output, label)
        clf_loss = loss.item()
        repr_loss = 0
        # Add the combinatorial objective
        #gc = GraphCut(metric = 'cosSim', lamda = 0.5)
        if comb_optim is not None:
            loss_comb = comb_optim(features, label)
            repr_loss = (1.0/ images.size(0)) * loss_comb.item()
            #print(loss_comb)
            loss += (1.0/ images.size(0)) * loss_comb

        #compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        f1_score = f1_loss(output, label)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        f1.update(f1_score.item(), images.size(0))
        curr_batch_time = time.time() - end
        batch_time.update(curr_batch_time)
        end = time.time()
        
        if wandb_var:
            logger = {'Train_steps': total_train_steps,
                    'Train_clf_loss': clf_loss,
                    'Train_repr_loss': repr_loss,
                    'Train_acc_1': acc1[0],
                    'Train_acc_5': acc5[0],
                    'Train_LR': get_lr(optimizer),
                    'Train_f1': f1_score,
                    'Train_batch_time': curr_batch_time 
                    }

            wandb.log(logger)        
        #Elapsed time
        

        if i % cfg.PRINT_FREQUENCY == 0:
            progress.print(i)
            if not writer == None:
                # add scalars to write
                writer.add_scalar('Train_loss', loss.item(), total_train_steps)
                writer.add_scalar('Train_acc_1', acc1[0], total_train_steps)
                writer.add_scalar('Train_acc_5', acc5[0], total_train_steps)
                writer.add_scalar('Train_LR', get_lr(optimizer),
                                  total_train_steps)

        total_train_steps += 1


def validate(val_loader, model, criterion, epoch, cfg, comb_optim, writer=None, wandb_var=False):
    '''
    Validate over one pass on the validation set
    '''
    step_count = int(len(val_loader)/ cfg.TEST.BATCH_SIZE)

    global total_val_steps
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':4.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    f1 = AverageMeter('f1', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             batch_time,
                             losses,
                             top1,
                             top5,
                             f1,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        clf_loss = 0
        repr_loss = 0
        total_images = 0
        for i in range(step_count):
            images, label = next(iter(val_loader))
            images = images[0].to(device)
            label = torch.flatten(torch.stack(label)).to(device)

            # compute output
            features, output = model(images)
            loss = criterion(output, label)
            clf_loss += (loss.item() * images.size(0))
            total_images += images.size(0)
            if comb_optim is not None:
                loss_comb = comb_optim(features, label)
                repr_loss = loss_comb.item()
                #print(loss_comb)
                loss += (1.0/images.size(0)) * loss_comb

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            f1_score = f1_loss(output, label)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            f1.update(f1_score.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQUENCY == 0:
                progress.print(i)
                if not writer == None:
                    # add scalars to write
                    writer.add_scalar('Val_loss', loss.item(), total_val_steps)
                    writer.add_scalar('Val_acc_1', acc1[0], total_val_steps)
                    writer.add_scalar('Val_acc_5', acc5[0], total_val_steps)
                    writer.add_scalar('Val_f1_score', f1_score, total_val_steps)

            total_val_steps += 1
        
        if wandb_var:
            logger = {'Epoch': epoch+1,
                    'Val_clf_loss': clf_loss/total_images,
                    'Val_repr_loss': repr_loss/total_images,
                    'Val_acc_1': top1.avg,
                    'Val_acc_5': top5.avg,
                    'Val_f1': f1.avg
                    }
            wandb.log(logger)
        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} F1 {f1_score.avg:.3f}'.format(
            top1=top1, top5=top5, f1_score=f1))

    return top1.avg


def resume_from_ckpt(ckpt_path, model, optimizer):
    '''
    Function to load a checkpoint file and resume from there
    '''
    checkpoint = torch.load(ckpt_path)
    # load model weights
    model.load_state_dict(checkpoint['state_dict'])
    # load optimizer weights
    optimizer.load_state_dict(checkpoint['optimizer'])

    logging.info("Loaded checkpoint from : {}".format(ckpt_path))
