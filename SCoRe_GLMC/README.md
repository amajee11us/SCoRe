# SCoRe - Submodular Combinatorial Representation Learning 
This repository aims to create a benchmark for submodular combinatorial loss functions for representation learning tasks. 
We benchmark widely adopted objective functions like contrastive loss, triplet loss, margin penalties etc. for image classification tasks against submodular combinatorial loss functions (added as regularizers). 

Our paper introduces a novel family of objective functions based on set-based submodular information measures. The paradigm shift in machine learning to adopt set-based information functions as learning objectives and exploiting their combinatorial properties to overcome inter-class bias and intra-class variance is the key motivation of SCoRe.

## Installation
The following packages are required to be installed before running training and evaluation operations.

a. Pytorch >= 1.8 ```conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch```

b. torchvision >=0.8.2 (install with torch)

c. easydict - ``` conda install -c conda-forge  easydict```

d. tensorboardx - ``` conda install -c conda-forge tensorboardx ```

e. pyyaml - ``` conda install pyyaml ```

f. numpy >= 1.18

g. dotenv - ```conda install -c conda-forge python-dotenv```

h. wandb - ```conda install -c conda-forge python-dotenv```

i. pytorch_metric_learning - ```conda install -c conda-forge pytorch-metric-learning```

Create the environment variables for accessing wandb.

a. Create a ```.env``` file in the root directory

b. Include the following details to access wandb APIs.
```
WANDB_API_KEY=<YOUR API KEY>
WANDB_USER_NAME=<YOUR USER NAME>
WANDB_PROJECT_NAME=SCoRe
```

For reproducing the results on a 64-bit Ubuntu machine with 2 RTX A6000 GPU an ```environment.yml``` file has been included for reference. 

Here we provide the results for SCoRe + GLMC to demonstrate the efficacy of objectives in SCoRe over the GLMC approach.

### Preparing Datasets
Download the datasets CIFAR-10, CIFAR-100, ImageNet, and iNaturalist18 to GLMC-2023/data. The directory should look like

````
GLMC-2023/data
├── CIFAR-100-python
├── CIFAR-10-batches-py
├── ImageNet
|   └── train
|   └── val
├── train_val2018
└── data_txt
    └── ImageNet_LT_val.txt
    └── ImageNet_LT_train.txt
    └── iNaturalist18_train.txt
    └── iNaturalist18_val.txt
    
````
## Training

for CIFAR-10-LT
````
python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4

python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4

python main.py --dataset cifar10 -a resnet32 --num_classes 10 --imbanlance_rate 0.1 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2 --label_weighting 1  --contrast_weight 6
````

for CIFAR-100-LT
````
python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4

python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2  --label_weighting 1.2  --contrast_weight 6

python main.py --dataset cifar100 -a resnet32 --num_classes 100 --imbanlance_rate 0.1 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.2  --label_weighting 1.2  --contrast_weight 6
````


for ImageNet-LT
````
python main.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 135 -b 120 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
````

for iNaturelist2018 
````
python main.py --dataset iNaturelist2018 -a resnext50_32x4d --num_classes 8142 --beta 0.5 --lr 0.1 --epochs 120 -b 128 --momentum 0.9 --weight_decay 1e-4 --resample_weighting 0.2 --label_weighting 1.0 --contrast_weight 10
````

## Testing
````
python test.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --resume model_path
````

## Results

|  | Method | CIFAR-10-LT |  |  | CIFAR-100-LT |  |  |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | IF=100 | 50 | 10 | 100 | 50 | 10 |
| 1 | CE (Baum & Wilczek, 1987) | 70.4 | 74.8 | 86.4 | 38.3 | 43.9 | 55.7 |
| 2 | BBN (Zhou et al., 2020) | 79.82 | 82.18 | 88.32 | 42.56 | 47.02 | 59.12 |
| 3 | CB-Focal (Cui et al., 2019b) | 74.6 | 79.3 | 87.1 | 39.6 | 45.2 | 58 |
| 4 | LogitAjust (Menon et al., 2021) | 80.92 | - | - | 42.01 | 47.03 | 57.74 |
| 5 | weight balancing (Alshammari et al., 2022) | - | - | - | 53.35 | 57.71 | 68.67 |
| 6 | Mixup (Zhang et al., 2018) | 73.06 | 77.82 | 87.1 | 39.54 | 54.99 | 58.02 |
| 7 | RISDA (Chen et al., 2022) | 79.89 | 79.89 | 79.89 | 50.16 | 53.84 | 62.38 |
| 8 | CMO (Park et al., 2022) | - | - | - | 47.2 | 51.7 | 58.4 |
| 9 | RIDE (3 experts) + CMO (Park et al., 2022) | - | - | - | 50 | 53 | 60.2 |
| 10 | RIDE (3 experts) (Wang et al., 2021) | - | - | - | 48.6 | 51.4 | 59.8 |
| 11 | KCL (Kang et al., 2021) | 77.6 | 81.7 | 88 | 42.8 | 46.3 | 57.6 |
| 12 | TSC (Li et al., 2022) | 79.7 | 82.9 | 88.7 | 42.8 | 46.3 | 57.6 |
| 13 | SSD (Li et al., 2021b) | - | - | - | 46.0 | 50.5 | 62.3 |
| 14 | BCL (Zhu et al., 2022) | 84.32 | 87.24 | 91.12 | 51.93 | 56.59 | 64.87 |
| 15 | PaCo (Cui et al., 2021) | 85.11 | 87.07 | 90.79 | 52.0 | 56.0 | 64.2 |
| 16 | PaCo + SCoRe-FL (ours) | $\mathbf{8 5 . 6 1}$ | $\mathbf{8 7 . 4 9}$ | $\mathbf{9 1 . 8 0}$ | $\mathbf{5 3 . 7 1}$ | $\mathbf{5 6 . 8 4}$ | $\mathbf{6 5 . 1 3}$ |
| 17 | GLMC (Du et al., 2023) | 88.50 | 91.04 | 94.90 | 58.0 | 63.78 | 73.43 |
| 18 | GLMC + SCoRe-GC (ours) | 89.38 | 90.32 | 94.67 | 60.01 | 63.16 | 73.50 |
| 19 | GLMC + SCoRe-FL (ours) | $\mathbf{9 2 . 3 3}$ | $\mathbf{9 3 . 8 7}$ | $\mathbf{9 4 . 9 3}$ | $\mathbf{6 1 . 3 3}$ | $\mathbf{6 4 . 9 0}$ | $\mathbf{7 3 . 7 8}$ |
