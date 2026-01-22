#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --batch_size 8 \
    --name_exp train_msd_task05 \
    --dataset_file msd \
    --fold 0 \
    --adaptformer_stages 2 3 \
    --prompt mask \
    --img_size 1024 \
    --sam2_version tiny \
    --epochs 5 \
    --data_root /data/datasets/jpg/MSD/Task05_Prostate \
    --task Task05 \
    --shots 3 \
    --lr 1e-4