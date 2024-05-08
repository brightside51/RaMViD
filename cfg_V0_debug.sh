#!/bin/bash
#
#SBATCH --partition=debug_8GB                                   # Partition (check with "$sinfo")
#SBATCH --qos=debug_8GB                                            # (Optional) 01.ctm-deep-05
#--container-image nvcr.io\#nvidia/pytorch:21.04-py3

#Commands / scripts to run (e.g., python3 train.py)
#$ enroot import docker://nvcr.io#nvidia/pytorch:21.04-py3
#$ enroot create --name container_name nvidia+pytorch+21.04-py3.sqsh

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 1 --microbatch 1 --seq_len 20 --max_num_mask_frames 4 --uncondition_rate 0.";
python3 scripts/video_train.py --data_dir ../../../../datasets/private/METABREST/T1W_Breast/video_data/V1/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
