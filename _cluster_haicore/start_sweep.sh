#!/bin/bash

# default config
sbatch resubmission.sh default 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Masking
# 60
sbatch resubmission.sh mask60 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.60 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 80
sbatch resubmission.sh mask80 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.80 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 90
sbatch resubmission.sh mask90 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.90 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Decoder depth
# 1
sbatch resubmission.sh decoder_depth1 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 1 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh decoder_depth2 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 2 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh decoder_depth4 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 4 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Number of attention Heads
# 8
sbatch resubmission.sh attention_heads8 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 8 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 16
sbatch resubmission.sh attention_heads16 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Attention dropout rate
sbatch resubmission.sh attention_dropout_rate0 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.0
sbatch resubmission.sh attention_dorpout_rate05 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.5

# Training length
# 2k epochs
sbatch resubmission.sh epochs2k 802 3d_fullres -tr EvaMAETrainer2kEpochs -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
# 4k epochs
sbatch resubmission.sh epochs4k 802 3d_fullres -tr EvaMAETrainer4kEpochs -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Learning Rate
# 3e-3
sbatch resubmission.sh lr3em3 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.003 --attention_drop_rate 0.2
# 3e-5
sbatch resubmission.sh lr3em5 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 16 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.00003 --attention_drop_rate 0.2

# Scaling experiments
sbatch resubmission.sh scaling_d20h12 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d24h12 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d28h12 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 12 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d20h16 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d24h16 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d28h16 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 16 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d20h24 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 20 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d24h24 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 24 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2
sbatch resubmission.sh scaling_d28h24 802 3d_fullres -tr EvaMAETrainer -num_gpus 4 --embed_dim 864 --batch_size 8 --mask_ratio 0.70 --encoder_eva_depth 28 --encoder_eva_numheads 24 --decoder_eva_depth 8 --decoder_eva_numheads 12 --initial_lr 0.0003 --attention_drop_rate 0.2

# Here you can add more configs you want to try or even just loop over some parameters
# e.g. for embed_dim in embed_dim_list etc.
