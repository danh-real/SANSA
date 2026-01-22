#!/bin/bash

# Inference script for Sarcoma dataset
# Usage: bash run_inference_sarcoma.sh

export CUDA_VISIBLE_DEVICES=0

EXP_NAME=eval_msd_btcv
DATA_ROOT=/data/datasets/jpg/BTCV
DATASET_FILE=btcv
CHECKPOINT=output/train_btcv/checkpoint0021.pth

for SHOT in 1 5;
do
  python inference_fss.py \
    --dataset_file $DATASET_FILE \
    --fold 0 \
    --resume $CHECKPOINT \
    --name_exp $EXP_NAME \
    --shot $SHOT \
    --adaptformer_stages 2 3 \
    --prompt mask \
    --img_size 1024 \
    --data_root $DATA_ROOT \
    --num_workers 8 \
    --threshold 0.5 \
    --device cuda \
    --sam2_version tiny
done