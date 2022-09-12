import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from PIL import Image
import numpy as np
import os


class CIFAR10Dataset(Dataset):
    def __init__(self,
                 data_path,
                 split,
                 transform=None,
                 num_classes=10,
                 random_seed=42,
                 download=True):
        '''
        Data loader for CIFAR10 dataset
        Arguments:
        data_path :
        '''
        self.split = split

        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        if not ("train" in self.split or "val" in self.split):
            raise Exception("Such split does not exist")

        self.data_path = data_path
        if not os.path.exists(self.data_path):
            raise Exception("Data path: {} does not exist for split {}".format(
                self.data_path, self.split))
        else:
            # No need to download thus override (since path exists)
            download = False
        _cifar10_obj = CIFAR10(root=self.data_path,
                               train=True if self.split == "train" else False,
                               transform=self.transform,
                               download=download)
        class_names = (
            'plane',
            'car',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        )
        self.classes = {i : class_names[i] for i in range(len(class_names))}

        # store as merged list
        data = _cifar10_obj.data
        labels = np.array(_cifar10_obj.targets)
        self.data_store = list(zip(data, labels))

        # generate random shuffled samples
        np.random.seed(random_seed)
        np.random.shuffle(self.data_store)

    def __getitem__(self, index):
        # Fetch an indexed entry in the image list
        img, label = self.data_store[index]

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.data_store)


# Test the loader
if __name__ == "__main__":
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32),  # square image transform
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    print('Unit test')
    cifar10_data = CIFAR10Dataset('/home/shared/anay/ton_618/data/cifar10',
                                  'train',
                                  transform=transformations)

    train_loader = torch.utils.data.DataLoader(dataset=cifar10_data,
                                               batch_size=10,
                                               shuffle=True)
    print(len(cifar10_data))
    x, y = next(iter(train_loader))
    print(x.shape, y.shape, [cifar10_data.classes[y_] for y_ in y.data.numpy()])