import yaml

import os
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import utils

import models.MagNet.attacks as attacks
import models.MagNet.mutation as mutation


# MagNet Reformer
from models.MagNet.defensive_models import (DefensiveModel1, DefensiveModel2,
                                            DefensiveModel3, DefensiveModel4)


from models.stargan.solver import Solver


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, noise_type, var=1.0):
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
        self.noise_type = noise_type
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

## NOTE: transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) 추가완료

def get_dataset(dataset, config=None):

    if dataset == 'mnist':
        train_set = datasets.MNIST('../data', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                   ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
        test_set = datasets.MNIST('../data', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

                                  ]))

    elif dataset == 'fashion-mnist':
        train_set = datasets.FashionMNIST('../data', train=True, download=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                          ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
        test_set = datasets.FashionMNIST('../data', train=False, download=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                         ]))

    elif dataset == 'cifar-10':
        train_set = datasets.CIFAR10('../data', train=True, download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                     ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])
        test_set = datasets.CIFAR10('../data/', train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ]))
    elif dataset == 'CelebA':
        train_set = CelebA(config.dataset_dir, config.attr_path, config.selected_attrs, 
                                     transform=transforms.Compose([
                                        T.RandomHorizontalFlip(),
                                        T.CenterCrop(config.celeba_crop_size),
                                        T.Resize(config.image_size),
                                        T.ToTensor(),
                                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                   ]), mode='train', noise_type=config.attack_type, var=config.train_noise_var)

        val_set = None
        test_set = CelebA(config.dataset_dir, config.attr_path, config.selected_attrs, 
                                     transform=transforms.Compose([
                                        T.CenterCrop(config.celeba_crop_size),
                                        T.Resize(config.image_size),
                                        T.ToTensor(),
                                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                   ]), mode='test', noise_type=config.attack_type, var=config.train_noise_var)
    else:
        raise ValueError("Undefined dataset")

    return train_set, val_set, test_set

def train_epoch_dae(model, train_loader, optimizer, lr_scheduler, weight_input_noise, weight_regularizer=0, device='cpu', attack_type='gaussian', data_augmentation=False):
    model.train()
    criterion = nn.MSELoss(reduction="mean")
    loss = 0
    for batch_idx, (X_mb, _) in tqdm(enumerate(train_loader), leave=False, desc='Mini-Batches',
                                     total=len(train_loader)):
        if data_augmentation:
        # if data_augmentation and batch_idx%2==0:
            for mini_batch_idx, img in enumerate(X_mb):
                params_list = [
                    [-3, -2, -1, 1, 2, 3],          # translation
                    [7, 8, 10, 11, 12,],            # scale
                    [-6, -5, -3, 3, 5, 6],          # shear
                    [5, 7, 9, 11, 13],              # contrast
                    [-50, -40, -30, 30, 40, 50],    # rotation
                    [-20, -10, 10, 20],             # brightness
                    # [1, 2, 3, 5, 7, 9],             # blur
                    # [1, 3, 5, 7, 9, 11],            # GaussianBlur
                    # [1, 3, 5],                      # MedianBlur
                    # [6, 9]                          # bilateraFilter
                    ]
                np_data = utils.to_numpy(img)
                np_data = np.transpose(np_data, (1, 2, 0))
                cho = random.randrange(0, len(params_list))
                par = random.choice(params_list[cho])
                np_mut_img = mutation.choose_fun(np_data, cho, par)
                np_mut_img = np.transpose(np_mut_img, (2, 0, 1))
                if mini_batch_idx == 0:
                    temp_np_X_mb = np.expand_dims(np_mut_img, axis=0)
                else:
                    temp_np_X_mb = np.append(temp_np_X_mb, np.expand_dims(np_mut_img, axis=0), axis=0)

            temp_X_mb = torch.tensor(temp_np_X_mb) 
            X_mb = torch.cat([X_mb, temp_X_mb], axis=0)
            X_mb = torch.clip(temp_X_mb, min=-1.0, max=1.0)

        if attack_type == 'none':
            X_noisy_mb = X_mb
            X_mb = X_mb.to(device)
            X_noisy_mb = X_noisy_mb.to(device)

        elif attack_type == 'gaussian':
            X_noisy_mb = X_mb + weight_input_noise * torch.randn(X_mb.shape)
            X_noisy_mb = torch.clamp(X_noisy_mb, min=0, max=1)
            X_mb, X_noisy_mb = X_mb.to(device), X_noisy_mb.to(device)
            X_mb = X_mb.to(device)
        elif attack_type == 'fgsm':
            black = np.zeros((X_mb.shape[0],X_mb.shape[1],X_mb.shape[2],X_mb.shape[3]))
            black = torch.FloatTensor(black).to(device)
            X_mb = X_mb.to(device)
            fgsm_attack = attacks.LinfPGDAttack(model=model, device=device, attack_type=attack_type, k=1)
            X_noisy_mb = attacks.perturb_batch(X_mb, black, model, fgsm_attack)
            X_noisy_mb = X_noisy_mb.to(device)
            X_pred_mb = model(X_mb)

        optimizer.zero_grad()
        X_pred_mb = model(X_noisy_mb)
        loss_mb = criterion(X_pred_mb, X_mb)
        loss_regularizer = 0

        for param in model.parameters():
            loss_regularizer += torch.linalg.norm(param)
        loss_mb_regularized = loss_mb + weight_regularizer * loss_regularizer
        loss += loss_mb_regularized.item()
        loss_mb.backward()
        optimizer.step()

    lr_scheduler.step()
    loss /= len(train_loader)

    return loss


def test_dae(model, data_loader, weight_input_noise, device='cpu', attack_type='gaussian'):

    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    loss = 0
    with torch.no_grad():
        for batch_idx, (X_mb, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Testing', leave=False):
            if attack_type == 'gaussian':
                X_noisy_mb = X_mb + weight_input_noise * torch.randn(X_mb.shape)
                X_noisy_mb = torch.clamp(X_noisy_mb, min=0, max=1)
                X_mb, X_noisy_mb = X_mb.to(device), X_noisy_mb.to(device)
                X_mb = X_mb.to(device)
            elif attack_type == 'fgsm':
                black = np.zeros((X_mb.shape[0],X_mb.shape[1],X_mb.shape[2],X_mb.shape[3]))
                black = torch.FloatTensor(black).to(device)
                X_mb = X_mb.to(device)
                fgsm_attack = attacks.LinfPGDAttack(model=model, device=device, attack_type=attack_type)
                X_noisy_mb = attacks.perturb_batch(X_mb, black, model, fgsm_attack)
                X_noisy_mb = X_noisy_mb.to(device)
                X_pred_mb = model(X_mb)
                loss_mb = criterion(X_pred_mb, X_mb)
                loss += loss_mb.item()
        loss /= len(data_loader)
    return loss


def main(config):
    seed = config.seed
    dataset = 'CelebA'
    defensive_model = config.defense_model_type
    num_epochs = config.reformer_num_epochs
    batch_size = config.reformer_batch_size
    lr = config.reformer_lr
    step_size = config.reformer_step_size
    gamma = config.reformer_gamma
    # weight_input_noise = config.reformer_weight_input_noise

    train_noise_var = config.train_noise_var
    test_noise_var = config.test_noise_var

    weight_regularizer = config.reformer_weight_regularizer
    interval_log_loss = config.reformer_interval_log_loss
    interval_log_images = config.reformer_interval_log_images
    interval_checkpoint = config.reformer_interval_checkpoint
    num_samples = config.reformer_num_samples

    np.random.seed(0)
    torch.manual_seed(seed)
    matplotlib.use('TkAgg')
    device = config.device
    print(f"Device: {device}")

    train_set, val_set, test_set = get_dataset(dataset, config=config)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    samples_idx = np.random.choice(len(test_loader), size=num_samples)
    sample_images = torch.stack([test_set[i][0] for i in samples_idx]).to(device)

    if defensive_model == 'defensive-model-1':
        model = DefensiveModel1()
    elif defensive_model == 'defensive-model-2':
        model = DefensiveModel2()
    elif defensive_model == 'defensive-model-3':
        model = DefensiveModel3()
    elif defensive_model == 'defensive-model-4':
        model = DefensiveModel4()
    else:
        raise ValueError("Undefined classifier")
    model = model.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                            step_size=step_size,
                                            gamma=gamma)

    config_dict = vars(config)

    for epoch_num in tqdm(range(num_epochs), leave=False, desc='Training Epochs:'):
        loss_train = train_epoch_dae(model, train_loader, optimizer, lr_scheduler, train_noise_var, weight_regularizer, device=config.device, attack_type=config.attack_type, data_augmentation=config.data_augmentation)
        # if epoch_num % interval_checkpoint == 0:
        #     loss_train_raw = test_dae(model, train_loader, device=config.device)
        #     # loss_val = test_dae(model, val_loader)
        #     loss_test = test_dae(model, test_loader, device=config.device)
        #     model_path = f'models/checkpoints/{dataset}_{defensive_model}_epoch-{epoch_num}_checkpoint.pth'
        #     model_checkpoint = {
        #         'epoch_num': epoch_num,
        #         'state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss_train': loss_train,
        #         'loss_train_raw': loss_train_raw,
        #         # 'loss_val': loss_val,
        #         'loss_test': loss_test
        #     }
            # torch.save(model_checkpoint, model_path)
        image_path = config.final_result_dir
        if epoch_num % interval_log_images == 0 and epoch_num != 0:
            
            sample_images_noisy_mb = sample_images + test_noise_var * torch.randn(sample_images.shape).to(device)
            sample_images_noisy_mb = torch.clamp(sample_images_noisy_mb, min=0, max=1)
            sample_images_noisy_mb = sample_images_noisy_mb.to(device)

            model.eval()
            sample_images_noisy_mb_reconstructed = model(sample_images_noisy_mb)
            utils.save_images(images=sample_images_noisy_mb_reconstructed, epoch=epoch_num, path=image_path)
        elif epoch_num == 0:
            utils.save_images(images=sample_images, epoch=epoch_num, path=image_path)
        elif epoch_num + 1 == range(num_epochs):

            sample_images_noisy_mb = sample_images + test_noise_var * torch.randn(sample_images.shape).to(device)
            sample_images_noisy_mb = torch.clamp(sample_images_noisy_mb, min=0, max=1)
            sample_images_noisy_mb = sample_images_noisy_mb.to(device)

            model.eval()
            sample_images_noisy_mb_reconstructed = model(sample_images_noisy_mb)
            utils.save_images(images=sample_images_noisy_mb_reconstructed, epoch=epoch_num, path=image_path)
            
        loss_train_raw = test_dae(model, train_loader, train_noise_var, device=config.device)
        loss_test = test_dae(model, test_loader, test_noise_var, device=config.device)

    model_path = f'{config.final_result_dir}/{dataset}_{defensive_model}_checkpoint.pth'
    model_checkpoint = {
        'epoch_num': epoch_num,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train,
        'loss_train_raw': loss_train_raw,
        # 'loss_val': loss_val,
        'loss_test': loss_test
    }
    torch.save(model_checkpoint, model_path)




if __name__ == '__main__':
    main()