import argparse
import os

def get_config(parser):

    # Model configuration.
    try:
        parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--batch_size', type=int, default=32, help='image batch size')  # 32
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    except argparse.ArgumentError:
        pass
        
    # Training configuration.
    try:
        parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--train_type', type=str, default='vanilla', choices=['vanilla', 'G_adv', 'Both_adv'])
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                            default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']),
    except argparse.ArgumentError:
        pass

    # Test configuration.
    try:
        parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--test_type', type=str, default='attack', choices=['vanilla', 'attack', 'feature_attack', 'conditional_attack'])
    except argparse.ArgumentError:
        pass

    # Miscellaneous.
    try:
        parser.add_argument('--num_workers', type=int, default=1)
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    except argparse.ArgumentError:
        pass

    # Directories.
    try:
        parser.add_argument('--celeba_image_dir', type=str, default='dataset/CelebA/images')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--attr_path', type=str, default='dataset/CelebA/list_attr_celeba.txt')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--log_dir', type=str, default='models/stargan/logs')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--model_save_dir', type=str, default='models/stargan/stargan_celeba_128/models')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--sample_dir', type=str, default='models/stargan/samples', help='stargan training sample dir')
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--test_result_dir', type=str, default='results', help='stargan test result dir')
    except argparse.ArgumentError:
        pass

    # Step size.
    try:
        parser.add_argument('--log_step', type=int, default=10)
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--sample_step', type=int, default=1000)
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--model_save_step', type=int, default=10000)
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--lr_update_step', type=int, default=1000)
    except argparse.ArgumentError:
        pass

    config = parser.parse_args()
    return config


def str2bool(v):
    return v.lower() in ('true')