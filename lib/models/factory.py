from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
factory object to host call to object classes for various vision models
Current Supported list:
1. AlexNet
2. VGG16 - Not implemented yet
NOTE: This is a rolling update
'''
import torch
import logging
from lib.utils import get_target_device

# Add model implementations here
from lib.models import alexnet as alexnet
from lib.models import vgg16 as vgg16

# Registry for models modelled as a dictionary
ARCH_REGISTRY = {"alexnet": alexnet.AlexNet, "vgg16": vgg16.VGG16}


def build_model(cfg):
    arch = cfg.ARCH  # fetch architecture

    if arch.lower() not in ARCH_REGISTRY.keys():
        logging.error("Model architecture not present in the registry.")
        return
    # We have the model. Register and return
    model = ARCH_REGISTRY.get(arch)(cfg)
    logging.info("Model loading done.")

    model.to(get_target_device(cfg))

    # Enable data parallal if GPU is found
    # TODO: test on multi-gpu setting
    if "gpu" in cfg.DEVICE:
        model = torch.nn.parallel.DataParallel(model, device_ids=cfg.GPU)

    return model
