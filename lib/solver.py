from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import logging


#Create the optimizer functions
def _adam_optimizer(cfg, model):
    params = model.parameters()

    lr = cfg.TRAIN.SOLVER_LR

    adam_optimizer = torch.optim.Adam(params=params, lr=lr)

    return adam_optimizer


def _sgd_optimizer(cfg, model):
    params = model.parameters()

    sgd_optimizer = torch.optim.SGD(params,
                                    cfg.TRAIN.SOLVER_LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    nesterov=cfg.TRAIN.NESTEROV)

    return sgd_optimizer


# define lr_scheduler functions
def _step_lr(cfg, optimizer):
    step_size = cfg.TRAIN.LR_DECAY_STEP
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=step_size,
                                           gamma=cfg.TRAIN.GAMMA)


def _multi_step(cfg, optimizer):
    step_size = cfg.TRAIN.LR_DECAY_MULTI_STEP
    return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=step_size,
                                                gamma=cfg.TRAIN.GAMMA)


def _explr(cfg, optimizer):
    pass


OPTIM_REGISTRY = {"adam": _adam_optimizer, "sgd": _sgd_optimizer}

LR_SHEDULER_REGISTRY = {
    "steplr": _step_lr,
    "multi_steplr": _multi_step,
    "exponential": _explr
}


def build_optimizer(cfg, model):
    option = cfg.TRAIN.OPTIMIZER

    if not model:
        logging.error(
            "Model parameters are required for optimizer to operate on.")
        raise Exception("Model not found.")

    if option.lower() not in OPTIM_REGISTRY.keys():
        logging.error("Optimizer not found.")
        raise NotImplementedError(
            "Optimizer with name {} nto found".format(option))

    optimizer = OPTIM_REGISTRY.get(option.lower())(cfg, model)

    return optimizer


def build_lr_scheduler(cfg, optimizer):
    option = cfg.TRAIN.LR_SCHEDULER

    if not optimizer:
        logging.error("Optimizer is required for optimizer to operate on.")
        raise Exception("Optimizer not found.")

    if option.lower() not in LR_SHEDULER_REGISTRY.keys():
        logging.error("Scheduler not found.")
        raise NotImplementedError(
            "Scheduler with name {} nto found".format(option))

    lr_scheduler = LR_SHEDULER_REGISTRY.get(option.lower())(cfg, optimizer)

    return lr_scheduler
