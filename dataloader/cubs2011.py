import os
import pandas as pd
import torch
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from torchvision import transforms


class CUBS(Dataset):
    
    def __init__(self, root, train=True, transform=None, loader=default_loader):        
        self.root = os.path.expanduser(root)
        self.base_folder = 'CUB_200_2011'
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.targets = self.data.loc[:,"target"].values.tolist()

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, 'images', row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, 'images', sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

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
    cubs2011_data = CUBS('/home/shared/anay/SupContrast/datasets',
                                  transform=transformations)

    train_loader = torch.utils.data.DataLoader(dataset=cubs2011_data,
                                               batch_size=1,
                                               shuffle=True)
    print(len(cubs2011_data))
    x, y = next(iter(train_loader))
    print(x)
    
    y = torch.flatten(torch.stack(y))
    print(y.shape)
    print(x.shape)
    print(y)