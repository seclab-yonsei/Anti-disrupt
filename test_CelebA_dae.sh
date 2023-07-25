#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
--gan_type stargan \
--defense_model_type ddpm \
--defense_noise none \
--attack_type pgd \
--test_noise_var 0.05 \
--save_image True \
# --detector \
# --data_augmentation \
# --detector \

# --result_path results 
# --defense_model_type defensive-model-1 \