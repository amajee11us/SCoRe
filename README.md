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

## Usage

You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to MedMNIST by `--dataset <dermamnist/organamnist>`.

**(1) Training on Cross-Entropy Loss**
```
python trainval_ce.py --batch_size 512 \
  --learning_rate 0.8 \
  --cosine --dataset <choice of dataset> 
```
**(2) Two Stage Learning Strategy in SCoRe**  
Stage 1: Training the feature extractor ResNet-50.
```
python train_stage1.py --batch_size 512 \
  <OTHER OPTIONS> \
  --method <choice of objective function> \
  --dataset <choice of dataset> 
```
Stage 2 : Training and evaluating the Linear model.
```
python trainval_stage2.py --batch_size 512 \
  --method <choice of objective function> \
  --dataset <choice of dataset> \
  --ckpt /path/to/model.pth \
  <OTHER OPTIONS>
```
Available choice of options common to stage 1 and stage 2 model training :
```
--print_freq PRINT_FREQ
                        print frequency
  --save_freq SAVE_FREQ
                        save frequency
  --batch_size BATCH_SIZE
                        batch_size
  --num_workers NUM_WORKERS
                        num of workers to use
  --epochs EPOCHS       number of training epochs
  --wandb WANDB         Boolean variable to indicate whether to use wandb for logging
  --learning_rate LEARNING_RATE
                        learning rate
  --lr_decay_epochs LR_DECAY_EPOCHS
                        where to decay lr, can be a list
  --lr_decay_rate LR_DECAY_RATE
                        decay rate for learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --momentum MOMENTUM   momentum
  --model MODEL
  --dataset {cifar10,cifar100,path,cubs,imagenet,imagenet32,organamnist,dermamnist,bloodmnist}
                        dataset
  --mean MEAN           mean of dataset in path in form of str tuple
  --std STD             std of dataset in path in form of str tuple
  --data_folder DATA_FOLDER
                        path to custom dataset
  --size SIZE           parameter for RandomResizedCrop
  --method {SupCon,SubmodSupCon,TripletLoss,SubmodTriplet,LiftedStructureLoss,NPairsLoss,MSLoss,SNNLoss,SubmodSNN,fl,gc,LogDet}
                        choose method
  --temp TEMP           temperature for loss function
  --use_imbalanced      using Class Imbalanced dataset
  --imbalance_ratio IMBALANCE_RATIO
                        Imbalance ratio for imbalanced data sampling
  --imbalance_classes IMBALANCE_CLASSES
                        List of class IDs in the range of [0, max_no_of_classes]
  --cosine              using cosine annealing
  --constant            using fixed LR
  --syncBN              using synchronized batch normalization
  --warm                warm-up for large batch training
  --trial TRIAL         id for recording multiple runs
  --resume_from RESUME_FROM
                        Checkpoint path to resume from.
```

## Results

| Objective Function                           | CIFAR-10 Longtail     | CIFAR-10 Step         | Organ-A-MNIST       | DermaMNIST                            |
|----------------------------------------------|-----------------------|-----------------------|---------------------|---------------------------------------|
| Cross-Entropy (CE)                           | 86.44                 | 74.49                 | 93.72               | 73.38                                 |
| Triplet Loss                                 | 85.94                 | 74.23                 | 88.59               | 70.85                                 |
| N-Pairs                                      | 89.70                 | 73.10                 | 90.71               | 71.82                                 |
| Lifted Structure Loss                        | 82.86                 | 73.98                 | 89.35               | 72.43                                 |
| SNN                                          | 83.65                 | 75.97                 | 89.42               | 72.67                                 |
| Multi-Similarity Loss                        | 82.40                 | 76.72                 | 91.64               | 71.94                                 |
| SupCon                                       | 89.96                 | 78.10                 | 93.13               | 73.51                                 |
| Submod-Triplet (ours)                        | 89.20                 | 74.36                 | 92.25               | 72.82                                 |
| Submod-SNN (ours)                            | 89.28                 | 78.76                 | 94.78               | 73.00                                 |
| Submod-SupCon (ours)                         | 90.81                 | 81.31                 | 96.39               | 73.79                                 |
| Graph-Cut [$S_{f}$] (ours)                   | 89.20                 | 76.89                 | 95.60               | 72.47                                 |
| Graph-Cut [$C_{f}$] (ours)                   | 90.83                 | 87.37                 | 96.55               | 73.56                                 |
| LogDet [$C_{f}$] (ours)                      | 90.80                 | 87.00                 | 95.98               | 73.11                                 |
| FL [ $C_{f}$ / $S_{f}$ ] (ours)                 | **91.80**             | **87.49**             | **96.87**           | **74.31**                             |


## Citation
Please feel free to cite our work when using this repo.
```

@InProceedings{score2024,
  title = 	 {{SC}o{R}e: Submodular Combinatorial Representation Learning},
  author =       {Majee, Anay and Kothawade, Suraj Nandkishor and Killamsetty, Krishnateja and Iyer, Rishabh K},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {34327--34349},
  year = 	 {2024},
  volume = 	 {235},
  publisher =    {PMLR}
}

```