def dae_train_parser(parser): 
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--reformer_num_epochs', type=int, default=100, help='')
    parser.add_argument('--reformer_batch_size', type=int, default=256, help='')    # 256
    parser.add_argument('--reformer_lr', type=float, default=0.001, help='')
    parser.add_argument('--reformer_step_size', type=int, default=10, help='')
    parser.add_argument('--reformer_gamma', type=float, default=0.8, help='')  # 0.95
    parser.add_argument('--reformer_weight_input_noise', type=float, default=0.1, help='')
    parser.add_argument('--reformer_weight_regularizer', type=float, default=1.e-9, help='')
    parser.add_argument('--reformer_interval_log_loss', type=int, default=1, help='')
    parser.add_argument('--reformer_interval_log_images', type=int, default=10, help='')
    parser.add_argument('--reformer_interval_checkpoint', type=int, default=25, help='')
    parser.add_argument('--reformer_num_samples', type=int, default=24, help='')
    
    # CelebA arguments
    parser.add_argument('--attr_path', type=str, default='dataset/CelebA/list_attr_celeba.txt')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                            default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    
    dae_config = parser.parse_args()
    return dae_config


def dae_test_parser(parser): 
    parser.add_argument('--reformer_batch_size', type=int, default=32, help='')   # 32, 1024
    parser.add_argument('--visualize', type=bool, default=True, help='')
    parser.add_argument('--visualization_path', type=str, default='visualizations', help='')

    dae_config = parser.parse_args()
    return dae_config


def dae_test_parser0(parser): 
    parser.add_argument('--reformer_batch_size', type=int, default=1024, help='')
    parser.add_argument('--visualize', type=bool, default=True, help='')
    parser.add_argument('--visualization_path', type=str, default='visualizations', help='')

    parser.add_argument('-batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--num_workers', type=int, default=1)

    # CelebA arguments
    parser.add_argument('--attr_path', type=str, default='dataset/CelebA/list_attr_celeba.txt')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                            default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    
    dae_config = parser.parse_args()
    return dae_config