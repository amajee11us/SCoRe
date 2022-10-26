# SCoRe - Submodular Combinatorial Representation Learning 
This repository aims to create a benchmark for submodular combinatorial loss functions for representation learning tasks. 
We benchmark widely adopted objective functions like contrastive loss, triplet loss, margin penalties etc. for image classification tasks against
submodular functions (added as regularizers). 

The aim of this experiment is to show that submodular functions lead to the formation of well-formed feature clusters with distinct decision boundaries for highly imbalanced real world datasets.

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

Create the environment variables for accessing wandb.

a. Create a ```.env``` file in the root directory

b. Include the following details to access wandb APIs.
```
WANDB_API_KEY=<YOUR API KEY>
WANDB_USER_NAME=<YOUR USER NAME>
WANDB_PROJECT_NAME=SCoRe
```

For reproducing the results on a 64-bit Ubuntu machine with a RTX A6000 GPU an ```environment.yml``` file has been included for reference.

## Objective Functions (TBD)
### Triplet Loss

### Supervised Contrastive Loss

### Cosine Similarity Objective

### Orthogonal Projection Loss

## Training Instructions
Run the below command to train a model using SCoRe.
```
python run.py --config_file configs/cifar_10_32x32.yaml
```
The evaluation of the model on the validation dataset will occur at the end of every epoch.

## Evaluation Instructions
TBD

## Results
The results for each experiment is tabulated below.

| Algorithm             | Config File | Loss Function |F1-score | Acc @ top1 | Acc @ top5 |
|---                    |:---:        |:---:          |:---:    |:---:       |:---:       |
|AlexNet_CIFAR-10_32x32 (baseline)| [link](configs/cifar_10_32x32.yaml)  | CE |         |71.17       |97.0        |
|AlexNet_CIFAR-10_32x32 | [link]()  | CE + CosSim         |       |        |

## References
TBD