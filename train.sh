#!/usr/bin/env bash
cd Codes

DATASET="ped2"
GPU="0"
#BATCH=4
BATCH=1
ITERS=10000

python train.py \
    -d $DATASET \
    -g $GPU \
    -b $BATCH \
    -i $ITERS \
    --train_folder ../Data/$DATASET/training/frames \
    --test_folder  ../Data/$DATASET/testing/frames
