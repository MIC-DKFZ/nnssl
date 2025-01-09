#!/bin/bash

# default config
sbatch -J default_743 resubmission.sh 743 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Masking
# 60
# sbatch -J mask60 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.60 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# # 80
# sbatch -J mask80 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.80 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# # 90
# sbatch -J mask90 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.90 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 
# # Decoder depth
# # 1
# sbatch -J decoder_depth1 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 1 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J decoder_depth2 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 2 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J decoder_depth4 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 4 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 
# # Number of attention Heads
# # 8
# sbatch -J attention_heads8 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 8 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# # 16
# sbatch -J attention_heads16 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 
# # Attention dropout rate
# sbatch -J attention_dropout_rate0 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.0
# sbatch -J attention_dorpout_rate05 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.5
# 
# # Training length
# # 2k epochs
# sbatch -J epochs2k resubmission.sh 802 3d_fullres -tr EvaMAETrainer2kEpochs -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# # 4k epochs
# sbatch -J epochs4k resubmission.sh 802 3d_fullres -tr EvaMAETrainer4kEpochs -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 
# # Learning Rate
# # 3e-3
# sbatch -J lr3em3 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.003 --attention_drop_rate 0.2
# # 3e-5
# sbatch -J lr3em5 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.00003 --attention_drop_rate 0.2
# 
# # Scaling experiments
# sbatch -J scaling_d20h12 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d24h12 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d28h12 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d20h16 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d24h16 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d28h16 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d20h24 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d24h24 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# sbatch -J scaling_d28h24 resubmission.sh 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Here you can add more configs you want to try or even just loop over some parameters
# e.g. for embed_dim in embed_dim_list etc.
