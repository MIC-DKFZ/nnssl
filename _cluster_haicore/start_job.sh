#!/bin/bash

# sbatch quick_resubmission.sh 743 3d_fullres -tr EvaMAETrainerDEBUG -num_gpus 4 --embed_dim 864 --batch_size 4 --mask_ratio 0.75 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003
sbatch resubmission.sh 743 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.75 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003

# Here you can add more configs you want to try or even just loop over some parameters
# e.g. for embed_dim in embed_dim_list etc.
