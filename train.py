######### 수정필요
import argparse
import json
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from tensorflow.python.client import device_lib
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import utils

# StarGAN
from models.stargan.data_loader import get_dataset, get_loader
from models.MagNet.config import dae_train_parser
from models.MagNet.train_defensive_model import main as dae_main
from models.stargan.config import get_config as stargan_get_config

# from models.MagNet.train_defensive_model import test_dae, train_epoch_dae
# from models.stargan.solver import Solver

## GANimation
# from models.ganimation.solver import Solver


if __name__ == '__main__':

    temp_parser = argparse.ArgumentParser()

    gpu_list = []
    for i in device_lib.list_local_devices():
        temp = i.name
        if 'GPU' in temp:
            gpu_list.append(int(temp[-1]))
    # Main configuration.
    ## gainmation 추가
    temp_parser.add_argument('--defense_model_type', type=str, default='defensive-model-1', choices=['defensive-model-1', 'defensive-model-2'], help='choose defense model that you want to train')
    temp_parser.add_argument('--attack_type', type=str, default='gaussian', choices=['gaussian', 'fgsm', 'i-fgsm', 'pgd', 'none'], help='choose noise you want to train')
    temp_parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'], help='choose dataset you want to use')
    temp_parser.add_argument('--train_noise_var', type=float, default=0.1, help='choose gaussian noise var for train')
    temp_parser.add_argument('--test_noise_var', type=float, default=0.1, help='choose gaussian noise var for test')
    temp_parser.add_argument('--gpu_num', type=int, default=1, choices=gpu_list, help='choose gpu to use')
    temp_parser.add_argument('--result_dir', type=str, default='results', help='')
    temp_parser.add_argument('--num_of_image', type=int, default=1, help='num of output images')
    temp_parser.add_argument('--stargan_adv', type=utils.str2bool, nargs='?', const=True, default=False, help="stargan is adv trained?")
    temp_parser.add_argument('--data_augmentation', type=utils.str2bool, nargs='?', const=True, default=False, help="do data augmentation")
    temp_config = temp_parser.parse_known_args()[0]

    temp_parser.add_argument('--dataset_dir', type=str, default=f'dataset/{temp_config.dataset}/images_a', help='')
    temp_parser.add_argument('--defensive_models_dir', type=str, default=f'{temp_config.result_dir}/{temp_config.defense_model_type}/{temp_config.dataset}', help='')
    temp_parser.add_argument('--device', type=str, default=torch.device('cuda:%d'%(temp_config.gpu_num) if torch.cuda.is_available() else 'cpu'), help='choose deivce to use')
    if temp_config.attack_type == 'gaussian':
        temp_parser.add_argument('--final_result_dir', type=str, default=f'{temp_config.result_dir}/{temp_config.defense_model_type}/{temp_config.dataset}/train/{temp_config.attack_type}', help='')
    else:
        temp_parser.add_argument('--final_result_dir', type=str, default=f'{temp_config.result_dir}/{temp_config.defense_model_type}/{temp_config.dataset}/train/{temp_config.attack_type}', help='')

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d"%(temp_config.gpu_num)

    utils.createFolder(temp_config.result_dir)

    if 'defensive-model' in temp_config.defense_model_type:
        config = dae_train_parser(temp_parser)

        config = stargan_get_config(temp_parser)
        
        if config.data_augmentation:
            config.final_result_dir += '_augmentation'

        config.attr_path='dataset/CelebA/images_a/list_attr_celeba.txt'
        config.celeba_image_dir='dataset/CelebA/images_a'

        print(config)
        dae_main(config=config)
        utils.save_config_dict(vars(config), os.path.join(config.final_result_dir, 'parameter.txt'))

