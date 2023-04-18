#!/bin/sh
BATCH_SIZE="36"
METHOD="SupCon"
TOTAL_EPOCHS_TRAIN="1000"

python main_supcon.py \
--batch_size $BATCH_SIZE --learning_rate 0.4 --temp 0.7 \
--cosine --method $METHOD --save_freq 100 --epochs $TOTAL_EPOCHS_TRAIN \
--dataset imagenet --size 224 --wandb True

python main_linear.py \
--batch_size $BATCH_SIZE --learning_rate 1 \
--ckpt save/SupCon/imagenet_models/SupCon_imagenet_resnet50_lr_0.5_decay_0.0001_bsz_${BATCH_SIZE}_temp_0.7_trial_0_cosine/last.pth \
--dataset imagenet --wandb True