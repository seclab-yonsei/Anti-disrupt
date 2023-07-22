import os
import argparse

import torch
from models.MagNet.config import dae_test_parser0 as dae_test_parser
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms as T
import random

## reformer
from models.MagNet.evaluate_defensive_model import get_defensive_model
from torch.utils import data
from torchvision.datasets import ImageFolder

import models.MagNet.attacks as attacks

import numpy as np

from tensorflow.python.client import device_lib
import utils


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, var=1.0):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.var = var

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))

        image = self.transform(image)
        label = torch.FloatTensor(label)

        # if self.noise_type == 'gaussian':
        #     image = add_gausian_noise(image, self.var)

        return image, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_CelebA_dataset(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, noise_type=None,
               train_noise_var=1.0, test_noise_var=1.0):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, 
                         noise_type=noise_type, var=train_noise_var if mode=='train' else test_noise_var)

    return dataset

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, )
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    gpu_list = []
    for i in device_lib.list_local_devices():
        temp = i.name
        if 'GPU' in temp:
            gpu_list.append(int(temp[-1]))

    parser.add_argument('--defense_model_type', type=str, default='defensive-model-1', choices=['defensive-model-1', 'defensive-model-2', 'defensive-model-3', 'defensive-model-4', 'defensive-model-5'], help='choose defense model that you want to train')
    parser.add_argument('--defense_noise', type=str, default='fgsm', choices=['gaussian', 'fgsm'], help='choose noise you want to train')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['gaussian', 'fgsm', 'i-fgsm', 'pgd'], help='choose noise you want to test')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'], help='choose dataset you want to use')
    # parser.add_argument('--train_noise_var', type=float, default=0.5, help='choose gaussian noise var for train')
    parser.add_argument('--test_noise_var', type=float, default=0.05, help='choose gaussian noise var for test')
    parser.add_argument('--gpu_num', type=int, default=0, choices=gpu_list, help='choose gpu to use')
    parser.add_argument('--result_dir', type=str, default='results', help='')
    parser.add_argument('--num_of_image', type=int, default=1, help='num of output images')
    
    config = parser.parse_known_args()[0]
    
    ## result/defense_model_type/dataset/gan_type/attack_type

    parser.add_argument('--dataset_dir', type=str, default=f'dataset/{config.dataset}/images', help='')
    parser.add_argument('--defensive_models_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/train/{config.defense_noise}', help='')
    parser.add_argument('--device', type=str, default=torch.device('cuda:%d'%(config.gpu_num) if torch.cuda.is_available() else 'cpu'), help='choose deivce to use')
    if config.attack_type == 'gaussian':
        parser.add_argument('--final_result_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/test/{config.attack_type}', help='')
    else:
        parser.add_argument('--final_result_dir', type=str, default=f'{config.result_dir}/{config.defense_model_type}/{config.dataset}/test/{config.attack_type}', help='')

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d"%(config.gpu_num)

    utils.createFolder(config.result_dir)
    

    if 'defensive-model' in config.defense_model_type:
        config = dae_test_parser(parser)
        print(config)
        utils.createFolder(config.final_result_dir)

        celeba_loader = get_loader(config.dataset_dir, config.attr_path, config.selected_attrs,
                            config.celeba_crop_size, config.image_size, config.batch_size,
                            'CelebA', config.mode, config.num_workers)

        ## defensive model type
        if config.defense_model_type!=None:
            if config.dataset == 'CelebA':
                data_loader = celeba_loader
        
            # Initialize Metrics
            l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
            n_dist, n_samples = 0, 0

            defensive_model = get_defensive_model(config.defensive_models_dir, config.defense_model_type, config.dataset, device=config.device)
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(config.device)

                if config.attack_type == 'gaussian':
                    X_noisy_mb = x_real + config.test_noise_var * torch.randn(x_real.shape).to(config.device)
                    X_noisy_mb = torch.clamp(X_noisy_mb, min=0, max=1)
                    x_real, X_noisy_mb = x_real.to(config.device), X_noisy_mb.to(config.device)
                elif config.attack_type == 'fgsm':
                    black = np.zeros((x_real.shape[0],x_real.shape[1],x_real.shape[2],x_real.shape[3]))
                    black = torch.FloatTensor(black).to(config.device)
                    x_real = x_real.to(config.device)
                    fgsm_attack = attacks.LinfPGDAttack(model=defensive_model, device=config.device, attack_type=config.attack_type)
                    X_noisy_mb = attacks.perturb_batch(x_real, black, defensive_model, fgsm_attack)
                    X_noisy_mb = X_noisy_mb.to(config.device)
                
                # X_noisy_mb = X_noisy_mb.to(device=config.device)
                x_ref = defensive_model(X_noisy_mb)
                x_real = defensive_model(x_real)
                
                # print(config.final_result_dir)
                utils.save_images(images=x_real, path=os.path.join(config.final_result_dir, 'original'))
                utils.save_images(images=X_noisy_mb, path=os.path.join(config.final_result_dir, 'noise'))
                utils.save_images(images=x_ref, path=os.path.join(config.final_result_dir, 'ref'))

                l1_error += F.l1_loss(x_ref, x_real)
                l2_error += F.mse_loss(x_ref, x_real)
                l0_error += (x_ref - x_real).norm(0)
                min_dist += (x_ref - x_real).norm(float('-inf'))
                n_samples += 1

                
            print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
            l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    