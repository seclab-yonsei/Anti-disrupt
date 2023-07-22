import os
import argparse

import torch
from tensorflow.python.client import device_lib

## defensive model 
from models.MagNet.config import dae_test_parser

## ganimation
from models.ganimation.main import main as ganimation_main
from models.ganimation.config import get_config as ganimation_get_config

## stargan
from models.stargan.main import main as stargan_main
from models.stargan.config import get_config as stargan_get_config

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    gpu_list = []
    for i in device_lib.list_local_devices():
        temp = i.name
        if 'GPU' in temp:
            gpu_list.append(int(temp[-1]))

    # Main configuration.
    parser.add_argument('--gan_type', type=str, default='stargan', choices=['stargan', 'stargan2', 'ganimation'], help='choose gan that you want to test')
    parser.add_argument('--defense_model_type', type=str, default='ddpm', choices=['none', 'defensive-model-1', 'defensive-model-2', 'adv-train-blur', 'adv-train', 'ddpm', 'both', 'diffpure'], help='choose defense model that you want to train')
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['none', 'fgsm', 'bim', 'pgd', 'none'], help='choose noise you want to test')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'], help='choose dataset you want to use')
    parser.add_argument('--defense_noise', type=str, default='none', choices=['none', 'gaussian', 'fgsm'], help='choose noise you want to train')
    # parser.add_argument('--train_noise_var', type=float, default=0.5, help='choose gaussian noise var for train')
    parser.add_argument('--test_noise_var', type=float, default=0.05, help='choose gaussian noise var for test')
    parser.add_argument('--gpu_num', type=int, default=0, choices=gpu_list, help='choose gpu to use')
    parser.add_argument('--save_image', type=utils.str2bool, nargs='?', const=True, default=True,
                        help="save images or not")
    parser.add_argument('--test_stable', type=utils.str2bool, nargs='?', const=True, default=False,
                        help="test stability or not(if True: ori_img -> dae(ori_img))")
    parser.add_argument('--result_dir', type=str, default='results', help='')
    parser.add_argument('--num_of_image', type=int, default=24, help='num of output images')
    parser.add_argument('--data_augmentation', type=utils.str2bool, nargs='?', const=True, default=False, help="do data augmentation") 
    parser.add_argument('--ddpm_start_num', type=int, default=10, help='')
    config = parser.parse_known_args()[0]

    if config.data_augmentation:
        parser.add_argument('--defensive_models_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/train/{config.defense_noise}_augmentation', help='')
    else:
        parser.add_argument('--defensive_models_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/train/{config.defense_noise}', help='')
    parser.add_argument('--final_result_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/{config.gan_type}/{config.defense_noise}/{config.attack_type}', help='')
    parser.add_argument('--device', type=str, default=torch.device('cuda:%d'%(config.gpu_num) if torch.cuda.is_available() else 'cpu'), help='choose deivce to use')


    if 'diffpure' in config.defense_model_type:
        config.ddpm_start_num = 500
    # utils.createFolder(config.final_result_dir)
    # path_list = [config.result_dir, config.dataset, config.defense_model_type]
    # result_dir=os.path.join(*path_list)
    # result_path = os.path.join(result_dir, '{}'.format(config.gan_type))
    # utils.createFolder(result_path)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d"%(config.gpu_num)

    # Model configuration.
    # dae: denoisingn auto encoder
    if 'defensive-model' in config.defense_model_type:
        config = dae_test_parser(parser)

    # config = dae_test_parser(parser)

    ## gan configuration
    if config.gan_type == 'stargan':
        config = stargan_get_config(parser)

        config.image_dir = 'dataset/CelebA/images_b'
        config.celeba_image_dir = 'dataset/CelebA/images_b'
        config.attr_path = 'dataset/CelebA/images_b/list_attr_celeba.txt'
        
        if config.data_augmentation:
            config.final_result_dir += '_augmentation'
        if config.defense_model_type == 'none':
            config.defense_model_type = None
        if config.attack_type == 'none':
            config.attack_type = None
        if config.test_result_dir!= config.final_result_dir: 
            config.test_result_dir=config.final_result_dir
        utils.print_config('stargan', config)
        stargan_main(config)
        utils.save_config_dict(vars(config), os.path.join(config.final_result_dir, 'parameter.txt'))


    elif config.gan_type == 'ganimation':
        config = ganimation_get_config(parser)
        if config.animation_results_dir!= config.final_result_dir: 
            config.animation_results_dir=config.final_result_dir
        utils.print_config('ganimation', config)
        ganimation_main(config)
        utils.save_config_dict(vars(config), os.path.join(config.final_result_dir, 'parameter.txt'))

    
    else: 
        print('please select correct gan_type')