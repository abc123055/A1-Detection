#!/usr/bin/env bash
cd Codes

DATASET="ped2"
GPU="0"
SNAPSHOT="checkpoints/ped2_l_2_alpha_1_lp_1.0_adv_0.05_gdl_1.0_flow_2.0/model.ckpt-10000"
EVALUATE="compute_video_accuracy"  # compute_auc | compute_frame_accuracy | compute_video_accuracy

python inference.py \
    -d $DATASET \
    -g $GPU \
    --snapshot_dir $SNAPSHOT \
    --test_folder  ../Data/$DATASET/testing/frames \
    --evaluate $EVALUATE