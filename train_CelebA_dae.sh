CUDA_VISIBLE_DEVICES=0 python train.py \
--defense_model_type ddpm \
--attack_type gaussian \
--train_noise_var 0.025 \
--test_noise_var 0.025 \
--gpu_num 0 \
--result_dir results \
--num_of_image 1 \
\
--reformer_num_epochs 100 \
--reformer_lr 0.001 \
--reformer_step_size 10 \
--reformer_gamma 0.8 \
--reformer_weight_input_noise 0.3 \
--reformer_weight_regularizer 1.e-9 \
--reformer_interval_log_loss 1 \
--reformer_interval_log_images 10 \
--reformer_interval_checkpoint 25 \
--reformer_num_samples 24 \
--reformer_batch_size 256

# --data_augmentation \
