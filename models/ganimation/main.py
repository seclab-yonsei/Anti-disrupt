import os

from models.ganimation.config import get_config
from models.ganimation.solver import Solver
from models.ganimation.data_loader import get_loader

from torch.backends import cudnn
import utils

## reformer
# from models.MagNet.evaluate_defensive_model import get_defensive_model



def main(config):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d"%(config.gpu_num)
    
    cudnn.benchmark = True  # Improves runtime if the input size is constant


    ## for Train
    config.outputs_dir = os.path.join('experiments', config.outputs_dir)
    config.log_dir = os.path.join(config.outputs_dir, config.log_dir)
    config.model_save_dir = os.path.join(config.outputs_dir, config.model_save_dir)
    config.sample_dir = os.path.join(config.outputs_dir, config.sample_dir)
    config.result_dir = os.path.join(config.outputs_dir, config.result_dir)

    data_loader = get_loader(image_dir=config.image_dir, 
                             attr_path=config.attr_path, 
                             c_dim=config.c_dim,
                             crop_size=config.crop_size, 
                             image_size=config.image_size, 
                             batch_size=config.batch_size, 
                             mode=config.mode,
                             num_workers=config.num_workers)
    
    config_dict = vars(config)
    solver = Solver(data_loader, config_dict)

    if config.mode == 'train':
        initialize_train_directories(config)
        solver.train()
    elif config.mode == 'animation':
        initialize_animation_directories(config)
        solver.animation(num=config.num_of_image)


def initialize_train_directories(config):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

def initialize_animation_directories(config):
    if not os.path.exists(config.animation_results_dir):
        os.makedirs(config.animation_results_dir)


# if __name__ == '__main__':
#     config = get_config()
#     main(config)