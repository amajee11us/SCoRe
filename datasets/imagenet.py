import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
import os


class ImageNetDataset(Dataset):
    def __init__(self,
                 data_path,
                 split,
                 transform=None,
                 num_classes=1000,
                 random_seed=42):
        '''
        Data loader for ImageNet dataset
        Arguments:
        data_path : 
        '''
        self.split = split

        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        if not ("train" in self.split or "val" in self.split):
            raise Exception("Such split does not exist")

        self.data_path = os.path.join(data_path, split)
        if not os.path.exists(self.data_path):
            raise Exception("Data path: {} does not exist for split {}".format(
                self.data_path, self.split))

        # Find list of classes and store
        class_names = [
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]
        class_names.sort()

        self.classes = {class_names[i]: i for i in range(len(class_names))}

        self.num_classes = len(class_names)

        # Find list of images and form image tuples
        self.data_store = []

        for cls in class_names:
            class_path = os.path.join(self.data_path, cls)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                #if self.to_tensor(Image.open(image_path)).shape[0] == 3: # only if an RGB image
                # add to data store
                self.data_store.append((image_path, self.classes[cls]))

        # generate random shuffled samples
        np.random.seed(random_seed)
        np.random.shuffle(self.data_store)

    def __getitem__(self, index):
        # Fetch an indexed entry in the image list
        image_path, label = self.data_store[index]

        #read image
        img = Image.open(image_path).convert("RGB")

        if not self.transform == None:
            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.data_store)


# Test the loader
if __name__ == "__main__":
    transformations = transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(200),  # square image transform
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    imagenet_data = ImageNetDataset('data/imagenet',
                                    'train',
                                    transform=transformations)

    train_loader = torch.utils.data.DataLoader(dataset=imagenet_data,
                                               batch_size=10,
                                               shuffle=True)
    img, label = imagenet_data.__getitem__(0)
    print(img.shape, label)