# SCoRe - Submodular Combinatorial Regularization 
This repository aims to create a benchmark for submodular combinatorial regularizers for representation learning tasks. 
We benchmark widely adopted objective functions like contrastive loss, triplet loss, margin penalties etc. for image classification tasks against
submodular functions (added as regularizers). 

The aim of this experiment is to show that submodular functions lead to the formation of well-formed feature clusters with distinct decision boundaries for highly imbalanced real world datasets.

## Installation
The following packages are required to be installed before running training and evaluation operations.

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
TBD

## References
TBD