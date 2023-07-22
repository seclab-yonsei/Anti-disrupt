from models.stargan.model import Generator, AvgBlurGenerator
from models.stargan.model import Discriminator
from torch.autograd import Variable

from torchvision.utils import save_image

import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import models.stargan.attacks as attacks
from attack import add_gausian_noise
import utils
from PIL import ImageFilter
from PIL import Image
from torchvision import transforms
import pickle

import pytorch_fid_wrapper as pfw

import models.stargan.defenses.smoothing as smoothing

## reformer
from models.MagNet.evaluate_defensive_model import get_defensive_model

from models.stargan.utils import save_images




# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config_dict):
        """Initialize configurations."""
    
        # NOTE: the following line create new class arguments with the
        # values in config_dict
        self.__dict__.update(**config_dict)

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        self.device = 'cuda:' + \
            str(self.gpu_num) if torch.cuda.is_available() else 'cpu'
        print(f"Model running on {self.device}")
        
        # Build the model and tensorboard.
        self.build_model()
        
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.G = AvgBlurGenerator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        if self.stargan_adv:
            self.model_save_dir = os.path.join(self.model_save_dir, 'adv')
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        
        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.load_model_weights(self.G, G_path)
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_and_restore_alt_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G2 = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D2 = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G2 = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D2 = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer2 = torch.optim.Adam(self.G2.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer2 = torch.optim.Adam(self.D2.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G2, 'G')
        self.print_network(self.D2, 'D')
            
        self.G2.to(self.device)
        self.D2.to(self.device)
        """Restore the trained generator and discriminator."""
        resume_iters = 50000
        model_save_dir = 'stargan/models'
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iters))
        
        # self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.load_model_weights(self.G2, G_path)
        self.D2.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def load_model_weights(self, model, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict, strict=False)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from models.stargan.logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Vanilla Training of StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)       # No Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, c_trg)       # No Attack
            out_src, out_cls = self.D(x_fake.detach())  # No Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)  # No Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                x_fake, _ = self.G(x_real, c_trg)     # No Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst, _ = self.G(x_fake, c_org)      # No Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
    
    def train_adv_gen(self):
        """Adversarial Training for StarGAN only for Generator, within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            pgd_attack = models.stargan.attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)       # No Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, c_trg)       # No Attack
            out_src, out_cls = self.D(x_fake.detach())  # No Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)  # No Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # Black image
            black = np.zeros((x_real.shape[0],3,256,256))
            black = torch.FloatTensor(black).to(self.device)
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                pgd_attack = models.stargan.attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
                x_real_adv = models.stargan.attacks.perturb_batch(x_real, black, c_trg, self.G, pgd_attack)

                x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_fake_adv = models.stargan.attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)
                x_reconst, _ = self.G(x_fake_adv, c_org)    # Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_adv_both(self):
        """G+D Adversarial Training for StarGAN with both Discriminator and Generator, within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # Black image
            black = np.zeros((x_real.shape[0],3,256,256))
            black = torch.FloatTensor(black).to(self.device)

            pgd_attack = models.stargan.attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)
            x_real_adv = models.stargan.attacks.perturb_batch(x_real, black, c_trg, self.G, pgd_attack)    # Adversarial training

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real_adv)   # Attack
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
            x_fake_adv = models.stargan.attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)    # Adversarial training
            out_src, out_cls = self.D(x_fake_adv.detach())  # Attack
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real_adv.data + (1 - alpha) * x_fake_adv.data).requires_grad_(True)  # Attack
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain
                x_fake, _ = self.G(x_real_adv, c_trg)   # Attack
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_fake_adv = models.stargan.attacks.perturb_batch(x_fake, black, c_org, self.G, pgd_attack)
                x_reconst, _ = self.G(x_fake_adv, c_org)    # Attack
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        elt, _ = self.G(x_fixed, c_fixed)
                        x_fake_list.append(elt)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Vanilla Training for StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset. No attack."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_attack(self, num=None, do_save_images=False, config=None):
        """Vanilla or blur attacks."""

        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
            self.image_dir=self.celeba_image_dir
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
            self.image_dir=self.rafd_image_dir
            
        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0
        
        # if self.attack_type!=None and self.attack_type!='gaussian':
        #         pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

        image_num = 0

        total_x_fake_list1 = [] # 원본 이미지
        total_x_fake_list2 = [] # 원본 딥페이크
        total_x_fake_list3 = [] # disrupt or reformed
        total_x_fake_list4 = [] # 공격 추가된 원본 이미지
        total_x_fake_list5 = [] # deepfake 원본 reform 

        accept_count = 0
        reject_count = 0

        for data_num, (x_real, c_org) in enumerate(data_loader):
            if num!=None and data_num == num and do_save_images:
                break   

            for num in range(len(x_real)):
                total_x_fake_list1.append(x_real[num].cpu())
            
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
            c_trg_tensor = torch.stack(c_trg_list, dim=0)
            
            # Translated images
            x_adv=None;x_ref=None
            attack_list = []

            ## TODO: attack type
            if self.attack_type=='gaussian': 
                x_adv=add_gausian_noise(x_real, self.test_noise_var)
            elif self.attack_type=='fgsm':
                black = np.zeros((x_real.shape[0],x_real.shape[1],x_real.shape[2],x_real.shape[3]))
                black = torch.FloatTensor(black).to(self.device)
                x_adv = x_real.to(self.device)
                fgsm_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, attack_type=self.attack_type, epsilon=self.test_noise_var)
                
                c_org = c_org.to(self.device)
                c_trg_tensor = c_trg_tensor.to(self.device)
                
                x_adv = attacks.perturb_batch(x_adv, black, c_org, self.G, fgsm_attack)
                x_adv = x_adv.to(self.device)

            elif self.attack_type=='i-fgsm':
                black = np.zeros((x_real.shape[0],x_real.shape[1],x_real.shape[2],x_real.shape[3]))
                black = torch.FloatTensor(black).to(self.device)
                x_adv = x_real.to(self.device)
                fgsm_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, attack_type=self.attack_type, epsilon=self.test_noise_var)
                
                c_org = c_org.to(self.device)
                c_trg_tensor = c_trg_tensor.to(self.device)
                
                x_adv = attacks.perturb_batch(x_adv, black, c_org, self.G, fgsm_attack)
                x_adv = x_adv.to(self.device)

                attack_list.append(x_adv)
                print(len(attack_list))

            elif self.attack_type=='pgd':
                for c_trg in c_trg_list:
                    # black, _ = self.G(x_real, c_trg)

                    black = np.zeros((x_real.shape[0],x_real.shape[1],x_real.shape[2],x_real.shape[3]))
                    black = torch.FloatTensor(black).to(self.device)

                    x_adv = x_real.to(self.device)
                    fgsm_attack = attacks.LinfPGDAttack(model=self.G, device=self.device, attack_type=self.attack_type, epsilon=self.test_noise_var)
                    
                    c_org = c_org.to(self.device)
                    c_trg_tensor = c_trg_tensor.to(self.device)
                    
                    x_adv = attacks.perturb_batch(x_adv, black, c_org, self.G, fgsm_attack)
                    x_adv = x_adv.to(self.device)
                    attack_list.append(x_adv)

            elif self.attack_type=='none' or self.attack_type==None:
                x_adv=x_real
    
            ## defensive model type
            if self.defense_model_type!=None:
                defensive_model = get_defensive_model(self.defensive_models_dir, self.defense_model_type, self.dataset, device=self.device)
                x_ref=defensive_model(x_adv)
            
            # x_fake_list1 = [x_real]
            # if x_adv!=None:
            #     x_fake_list2 = [x_adv]
            # else: x_fake_list2=None
            # if x_ref!=None:
            #     x_fake_list3 = [x_ref]
            # else: x_fake_list3=None  

            x_fake_list1 = []
            if x_adv!=None:
                x_fake_list2 = []
            else: x_fake_list2=None
            if x_ref!=None:
                x_fake_list3 = []
            else: x_fake_list3=None      
            
            for i in range(len(c_trg_list) + 1):
                x_fake_list1.append([])
                if x_adv!=None:
                    x_fake_list2.append([])
                else: x_fake_list2=None
                if x_ref!=None:
                    x_fake_list3.append([])
                else: x_fake_list3=None 

            # x_fake_list1[0].append(x_real)

             # 원본 이미지 reform
            if self.test_stable:
                x_real = defensive_model(x_real)
            x_temp_list = list()

            for i in range(len(x_real)):
                
                x_fake_list1[0].append(x_real[i])
                # x_fake_list1[0].append(self.denorm(x_real[i]))
            if x_adv!=None:
                # x_fake_list2[0].append(x_adv)
                for i in range(len(x_adv)):
                        x_fake_list2[0].append(x_adv[i])
                        # x_fake_list2[0].append(self.denorm(x_adv[i]))
            else: x_fake_list2=None
            if x_ref!=None:
                # x_fake_list3[0].append(x_ref)
                for i in range(len(x_ref)):
                        x_fake_list3[0].append(x_ref[i])
                        # x_fake_list3[0].append(self.denorm(x_ref[i]))
            else: x_fake_list3=None  
            #  
            # utils.save_images(images=x_ref, path='./ref')

            for idx, c_trg in enumerate(c_trg_list):
                
                x_adv = attack_list[idx]

                # print('image', i, 'class', idx)
                with torch.no_grad():

                    x_real_mod = x_real

                    # 원본 reformer 통과
                    # x_real_mod = defensive_model(x_real_mod)

                    # x_real_mod = self.blur_tensor(x_real_mod) # use blur
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)
                
                ## append original img & original output
                for i in range(len(gen_noattack)):
                    x_fake_list1[idx+1].append(gen_noattack[i])
                    # x_fake_list1[idx+1].append(self.denorm(gen_noattack[i]))
                # x_fake_list1[idx+1].append(gen_noattack)
                '''# Attacks
                # x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)                          # Vanilla attack
                # x_adv, perturb, blurred_image = pgd_attack.perturb_blur(x_real, gen_noattack, c_trg)    # White-box attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_iter_full(x_real, gen_noattack, c_trg)         # Spread-spectrum attack on blur
                # x_adv, perturb = pgd_attack.perturb_blur_eot(x_real, gen_noattack, c_trg)               # EoT blur adaptation

                # Generate adversarial example
                # x_adv = x_real + perturb

                # No attack
                # x_adv = x_real

                # x_adv = self.blur_tensor(x_adv)   # use blur'''
                
                if x_fake_list2!=None:
                    with torch.no_grad():
                        x_adv_mod = x_adv
                        # x_adv_mod = self.blur_tensor(x_adv_mod) # use blur
                        gen_adv, gen_noattack_feats = self.G(x_adv_mod, c_trg)
                    for i in range(len(gen_adv)):
                        x_fake_list2[idx+1].append(gen_adv[i])
                        # x_fake_list2[idx+1].append(self.denorm(gen_adv[i]))
                    # x_fake_list2[idx+1].append(gen_adv)
                    # x_fake_list2.append(gen_adv)

                if config.detector and self.defense_model_type!=None:  
                    diff = torch.abs(x_real - defensive_model(x_real))
                    marks = torch.mean(torch.pow(diff, 1), axis=(1,2,3))
                    # sorted_marks, _ = torch.sort(marks)
                    threshold = marks.mean().item() * 0.7


                if x_fake_list3!=None:
                    with torch.no_grad():
                        if config.detector:
                            diff = torch.abs(x_adv - x_ref)
                            marks = torch.mean(torch.pow(diff, 2), axis=(1,2,3))

                            new_gen_noattack = []
                            new_x_ref = []
                            new_c_trg = []

                            for m_num in range(len(x_ref)):
                                if marks[m_num] < threshold:
                                    new_gen_noattack.append(gen_noattack[m_num])
                                    new_x_ref.append(x_ref[m_num])
                                    new_c_trg.append(c_trg[m_num])
                                    accept_count += 1
                                else:
                                    reject_count += 1

                            gen_noattack = torch.stack(new_gen_noattack, dim=0)
                            new_x_ref = torch.stack(new_x_ref, dim=0)
                            new_c_trg = torch.stack(new_c_trg, dim=0)

                            gen_ref, _ = self.G(new_x_ref, new_c_trg)
                        else:
                            gen_ref, _ = self.G(x_ref, c_trg)
                    
                    # utils.save_images(self.denorm(gen_ref), path='temp_ref')

                    for i in range(len(gen_ref)):
                        x_fake_list3[idx+1].append(gen_ref[i])
                        # x_fake_list3[idx+1].append(self.denorm(gen_ref[i]))
                    # x_fake_list3[idx+1].append(self.denorm(gen_ref))
                    # x_fake_list3.append(gen_ref)

                ## TODO: 여기도 뭐랑 비교할지(original vs. result) 파라미터 받는것도 좋을듯?
                ## 그럼 이미지 저장도 비교군만 저장되게 바꿀까?
                ## save_img1, 2, 3 만들어서 original vs result를 저장해도 될 듯 ~~
                    
                ## Compare(reformer 전후)
                ## resulting_images_reg_ref vs resulting_images_reg_xadv

                # original=x_real_mod
                original = gen_noattack

                if self.defense_model_type==None:
                    result = gen_adv
                else:   # defensive 모델 존재하면
                    result = gen_ref

                
                l1_error += F.l1_loss(result, original)
                l2_error += F.mse_loss(result, original)
                l0_error += (result - original).norm(0)
                min_dist += (result - original).norm(float('-inf'))
                
                if F.mse_loss(result, original) > 0.05:
                    n_dist += 1
                n_samples += 1

                if do_save_images:
                    for num in range(len(gen_noattack)):
                        total_x_fake_list2.append(gen_noattack[num].cpu())
                    for num in range(len(result)):
                        total_x_fake_list3.append(result[num].cpu())
                    for num in range(len(x_adv_mod)):
                        total_x_fake_list4.append(x_adv_mod[num].cpu())
                    # for num in range(len(x_real)):
                    #     with torch.no_grad():
                    #         x_ori_ref=defensive_model(x_real)
                    #         total_x_fake_list5.append(x_ori_ref[num].cpu())

            if do_save_images:
                # Save the translated images.
                if x_fake_list1!=None:
                    # x_concat = torch.cat(x_fake_list1, dim=3)
                    path=os.path.join(self.final_result_dir, 'real')
                    utils.createFolder(path)
                    # save_image(self.denorm(x_concat.data.cpu()), os.path.join(path,f'{i+1}-images.jpg'), nrow=1, padding=0)
                    temp_image_num = save_images(images=x_fake_list1, path=path, image_num=image_num)
                
                if x_fake_list2!=None:
                    # x_concat = torch.cat(x_fake_list2, dim=3)
                    path=os.path.join(self.final_result_dir, 'adv')
                    utils.createFolder(path)
                    # save_image(self.denorm(x_concat.data.cpu()), os.path.join(path,'{}-images.jpg'.format(i+1)), nrow=1, padding=0)
                    temp_image_num = save_images(images=x_fake_list2, path=path, image_num=image_num)
                
                if x_fake_list3!=None:
                    # x_concat = torch.cat(x_fake_list3, dim=3)
                    path=os.path.join(self.final_result_dir, 'ref')
                    utils.createFolder(path)
                    ## TODO: reformer 다시 train 필요
                    temp_image_num = save_images(images=x_fake_list3, path=path, image_num=image_num)
                    # save_image(self.denorm(x_concat.data.cpu()), os.path.join(path,'{}-images.jpg'.format(i+1)), nrow=1, padding=0)
                    # save_image(x_concat.data.cpu(), os.path.join(path,'{}-images.jpg'.format(i+1)), nrow=1, padding=0)
                

                # total_x_fake_list3.append()

                image_num = temp_image_num


                # if i == num:     # stop after this many images
                #     break

        
        if do_save_images:
            with open('./dataset/CelebA/pickle/original.pkl', 'wb') as f:
                pickle.dump(total_x_fake_list1, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
            with open('./dataset/CelebA/pickle/deepfake.pkl', 'wb') as f:
                pickle.dump(total_x_fake_list2, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open('./dataset/CelebA/pickle/disrupt@reform.pkl', 'wb') as f:
                pickle.dump(total_x_fake_list3, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open('./dataset/CelebA/pickle/adversarial.pkl', 'wb') as f:
                pickle.dump(total_x_fake_list4, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open('./dataset/CelebA/pickle/original_reform.pkl', 'wb') as f:
                pickle.dump(total_x_fake_list5, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Print metrics
        # print(result,' vs. ',original)
        
        pfw.set_config(batch_size=original.shape[0], device=config.device)
        fid = pfw.fid(original, result)
        print(f'FID: {fid}')
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))
        print(f'Reject Rate {reject_count/(accept_count+reject_count)}')


    def test_attack_feats(self):
        """Feature-level attacks"""

        # Mapping of feature layers to indices
        layer_dict = {0: 2, 1: 5, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12, 7: 13, 8: 14, 9: 17, 10: 20, 11: None}

        for layer_num_orig in range(12):    # 11 layers + output
            # Load the trained generator.
            self.restore_model(self.test_iters)
            
            # Set data loader.
            if self.dataset == 'CelebA':
                data_loader = self.celeba_loader
            elif self.dataset == 'RaFD':
                data_loader = self.rafd_loader

            # Initialize Metrics
            l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
            n_dist, n_samples = 0, 0

            print('Layer', layer_num_orig)

            for i, (x_real, c_org) in enumerate(data_loader):
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                layer_num = layer_dict[layer_num_orig]  # get layer number
                pgd_attack = models.stargan.attacks.LinfPGDAttack(model=self.G, device=self.device, feat=layer_num)

                # Translate images.
                x_fake_list = [x_real]

                for c_trg in c_trg_list:
                    with torch.no_grad():
                        gen_noattack, gen_noattack_feats = self.G(x_real, c_trg)

                    # Attack
                    if layer_num == None:
                        x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)
                    else:
                        x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack_feats[layer_num], c_trg)
                        
                    x_adv = x_real + perturb

                    # Metrics
                    with torch.no_grad():
                        gen, gen_feats = self.G(x_adv, c_trg)

                        # Add to lists
                        x_fake_list.append(x_adv)
                        x_fake_list.append(gen)

                        l1_error += F.l1_loss(gen, gen_noattack)
                        l2_error += F.mse_loss(gen, gen_noattack)
                        l0_error += (gen - gen_noattack).norm(0)
                        min_dist += (gen - gen_noattack).norm(float('-inf'))
                        if F.mse_loss(gen, gen_noattack) > 0.05:
                            n_dist += 1
                        n_samples += 1

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-{}-images.jpg'.format(layer_num_orig, i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                if i == 49:
                    break
            
            # Print metrics
            print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
            l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))


    def test_attack_cond(self):
        """Class conditional transfer"""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Initialize Metrics
        l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
        n_dist, n_samples = 0, 0

        for i, (x_real, c_org) in enumerate(data_loader):
            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            pgd_attack = models.stargan.attacks.LinfPGDAttack(model=self.G, device=self.device, feat=None)

            # Translate images.
            x_fake_list = [x_real]
            
            for idx, c_trg in enumerate(c_trg_list):
                print(i, idx)
                with torch.no_grad():
                    x_real_mod = x_real
                    gen_noattack, gen_noattack_feats = self.G(x_real_mod, c_trg)

                # Transfer to different classes
                if idx == 0:
                    # Wrong Class
                    x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg_list[0])

                    # Joint Class Conditional
                    # x_adv, perturb = pgd_attack.perturb_joint_class(x_real, gen_noattack, c_trg_list)

                    # Iterative Class Conditional
                    # x_adv, perturb = pgd_attack.perturb_iter_class(x_real, gen_noattack, c_trg_list)
                    
                # Correct Class
                # x_adv, perturb = pgd_attack.perturb(x_real, gen_noattack, c_trg)

                x_adv = x_real + perturb

                # Metrics
                with torch.no_grad():
                    gen, _ = self.G(x_adv, c_trg)

                    # Add to lists
                    x_fake_list.append(x_adv)
                    # x_fake_list.append(perturb)
                    x_fake_list.append(gen)

                    l1_error += F.l1_loss(gen, gen_noattack)
                    l2_error += F.mse_loss(gen, gen_noattack)
                    l0_error += (gen - gen_noattack).norm(0)
                    min_dist += (gen - gen_noattack).norm(float('-inf'))
                    if F.mse_loss(gen, gen_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            if i == 49:
                break
        
        # Print metrics
        print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
        l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples, min_dist / n_samples))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def blur_tensor(self, tensor):
        # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=9).to(self.device)
        preproc = smoothing.GaussianSmoothing2D(sigma=1.5, channels=3, kernel_size=11).to(self.device)
        return preproc(tensor)
