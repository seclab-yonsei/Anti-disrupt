#!/bin/bash

python make_fig.py \
--gan_type stargan \
--defense_model_type defensive-model-1 \
--attack_type gaussian \
--defense_noise fgsm \


# --result_path results 
# --defense_model_type defensive-model-4 \