import os
import argparse

try:
    from models.stargan.solver import Solver
    from models.stargan.data_loader import get_loader
except ModuleNotFoundError:
    from solver import Solver
    from data_loader import get_loader

from torch.backends import cudnn

import utils


def main(config):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d"%(config.gpu_num)

    cudnn.benchmark = True

    # Create directories if not exist.
    # if not os.path.exists(config.log_dir):
    #     os.makedirs(config.log_dir)
    # if not os.path.exists(config.model_save_dir):
    #     os.makedirs(config.model_save_dir)
    # if not os.path.exists(config.sample_dir):
    #     os.makedirs(config.sample_dir)
    # if not os.path.exists(config.test_result_dir):
    #     os.makedirs(config.test_result_dir)
        


    # Data loader.
    celeba_loader = None
    rafd_loader = None



    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    
    config_dict = vars(config)
    
    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config_dict)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            # Vanilla training
            if config.train_type == 'vanilla':
                solver.train()
            # Generator adversarial training
            elif config.train_type == 'G_adv':
                solver.train_adv_gen()
            # G+D adversarial training
            elif config.train_type == 'Both_adv':
                solver.train_adv_both()

        elif config.dataset in ['Both']:
            solver.train_multi()      
                  
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            # Normal inference
            if config.test_type == 'vanilla':
                solver.test()
            ## Attack inference (here)
            elif config.test_type == 'attack':
                solver.test_attack(num=config.num_of_image, do_save_images=config.save_image, config=config)
            # Feature attack experiment
            elif config.test_type == 'feature_attack':
                solver.test_attack_feats()
            # Conditional attack experiment
            elif config.test_type == 'conditional_attack':
                solver.test_attack_cond()
    
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=14, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=50000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=250000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=50000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=utils.str2bool, default=False)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='./dataset/CelebA/images_a')
    parser.add_argument('--attr_path', type=str, default='./dataset/CelebA/images_a/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='adv/logs')
    parser.add_argument('--model_save_dir', type=str, default='adv/models')
    parser.add_argument('--sample_dir', type=str, default='adv/samples')
    parser.add_argument('--result_dir', type=str, default='adv/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    parser.add_argument('--gpu_num', type=int, default=0, help='choose gpu to use')
    parser.add_argument('--test_noise_var', type=float, default=0.1, help='choose gaussian noise var for test')
    parser.add_argument('--train_type', type=str, default='Both_adv')   # vanilla, Both_adv
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['none', 'fgsm', 'bim', 'pgd', 'none'], help='choose noise you want to test')

    config = parser.parse_args()
    print(config)
    main(config)