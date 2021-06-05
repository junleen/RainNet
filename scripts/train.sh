#!/usr/bin/env bash
python train.py \
--dataset_root ../dataset/iHarmony4/ \
--name experiment_train \
--model rainnet \
--netG rainnet \
--dataset_mode iharmony4 \
--is_train 1 \
--gan_mode wgangp \
--normD instance \
--normG RAIN \
--preprocess none \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size 11 \
--num_threads 6 \
--lambda_L1 100 \
--print_freq 400 \
--gpu_ids 0 \
#--continue_train \
#--load_iter 87 \
#--epoch 88 \
