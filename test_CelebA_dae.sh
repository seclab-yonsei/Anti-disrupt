#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python test.py \
--gan_type stargan \
--defense_model_type ddpm \
--defense_noise none \
--attack_type pgd \
--test_noise_var 0.05 \
--test_stable False \
--gpu_num 0 \
--save_image True \
--num_of_image 24 \
# --detector \
# --data_augmentation \
# --detector \

# --result_path results 
# --defense_model_type defensive-model-1 \