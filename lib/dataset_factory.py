from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
Data-store

1. Load and validate transforms (boolean transforms)
2. load dataset object
3. Apply transforms
4. create an iterable dataloader
'''
import torch
import torchvision.transforms as transforms

# in-built dataset
from datasets.cifar10 import CIFAR10Dataset
from datasets.imagenet import ImageNetDataset

import logging
DATASET_REGISTRY = ["imagenet", "cifar10", "cifar100"]


def load_transforms(cfg, split):
    '''
    List of all possible transforms

    NOTE: Transforms happen in order so do not change the order in this script
    '''
    transform_list = []  # this will get populated with objects

    dataset_params = cfg.TEST
    if "train" in split:
        dataset_params = cfg.TRAIN

    # Now the actual transforms
    if dataset_params.DATASET.PIL:
        transform_list.append(transforms.ToPILImage())

    if dataset_params.DATASET.RESIZE:
        transform_list.append(transforms.Resize(dataset_params.IMAGE_SIZE))
    if dataset_params.DATASET.RANDOM_CROP:
        transform_list.append(
            transforms.RandomResizedCrop(dataset_params.DATASET.CROP_SIZE))
    if dataset_params.DATASET.RANDOM_FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())
    if dataset_params.DATASET.CENTER_CROP:
        transform_list.append(
            transforms.CenterCrop(dataset_params.DATASET.CENTER_CROP_SIZE))

    # Now the image transforms
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(mean=dataset_params.DATASET.NORM_MEAN,
                             std=dataset_params.DATASET.NORM_STD_DEV))

    logging.info("\nTransforms used :\n{}".format(transform_list))
    return transforms.Compose(transform_list)


def load_and_apply(cfg, transforms, split):
    '''
    Apply transforms to the dataset object and return the consolidated iterable list
    '''
    dataset_params = cfg.TEST
    if "train" in split:
        dataset_params = cfg.TRAIN

    if "imagenet" in dataset_params.DATASET.NAME:
        dataset = ImageNetDataset(data_path=dataset_params.DATA_DIR,
                                  split=split,
                                  transform=transforms)
    elif "cifar10" in dataset_params.DATASET.NAME:
        dataset = CIFAR10Dataset(data_path=dataset_params.DATA_DIR,
                                 split=split,
                                 transform=transforms)
    else:
        logging.error(
            "Dataset not found. Please provide correct dataset values")
        raise Exception("Invalid dataset name provided.")

    return dataset


def build_dataset(cfg, split='train'):
    '''verify dataset'''
    if cfg.TRAIN.DATASET.NAME not in DATASET_REGISTRY:
        logging.error("Dataset with name {} not found.".format(
            cfg.TRAIN.DATASET.NAME))
        raise Exception("Dataset not found.")
    trnsfrm = load_transforms(cfg, split)
    data_object = load_and_apply(cfg, trnsfrm, split)
    logging.info(
        "Dataset created: {} split \n \t\tTOTAL_COUNT : {} \n \t\tNUM_CLASSES: {}"
        .format(split, len(data_object), cfg.TRAIN.NUM_CLASSES))

    batch_size = cfg.TRAIN.BATCH_SIZE if "train" in split else cfg.TEST.BATCH_SIZE

    #torch iterable loader
    loader = torch.utils.data.DataLoader(data_object,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=cfg.NUM_WORKERS,
                                         drop_last=True,
                                         batch_size=batch_size)
    return loader
