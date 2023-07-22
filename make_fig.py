import os
import argparse

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
# import matplotlib.image as img

import cv2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Main configuration.
    parser.add_argument('--gan_type', type=str, default='stargan', choices=['stargan', 'stargan2', 'ganimation'], help='choose gan that you want to test')
    parser.add_argument('--defense_model_type', type=str, default='defensive-model-1', choices=['none', 'defensive-model-1', 'defensive-model-2', 'defensive-model-3', 'defensive-model-4', 'defensive-model-5'], help='choose defense model that you want to train')
    parser.add_argument('--attack_type', type=str, default='gaussian', choices=['gaussian', 'fgsm', 'i-fgsm', 'pgd', 'none'], help='choose noise you want to test')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'], help='choose dataset you want to use')
    parser.add_argument('--defense_noise', type=str, default='gaussian', choices=['gaussian', 'fgsm', 'i-fgsm', 'pgd'], help='choose noise you want to train')



    config = parser.parse_known_args()[0]

    parser.add_argument('--final_result_dir', type=str, default=f'figures/{config.defense_model_type}/{config.dataset}/{config.gan_type}/{config.attack_type}', help='')

    config = parser.parse_args()

    folder_path = (f'./results/{config.defense_model_type}/{config.dataset}/{config.gan_type}/{config.defense_noise}/{config.attack_type}/ref')
    
    file_list = os.listdir(folder_path)
    image_count = len(file_list)
    attr_list = os.listdir(os.path.join(folder_path, '0'))
    attr_count = len(attr_list)

    fig = plt.figure()
    rows = 5
    cols = 6

    count = 0
    for i in range(image_count):
        for j in range(attr_count):
            image_path = os.path.join(os.path.join(folder_path, str(i)), f'{j}.jpg')
            # img_test = img.imread(image_path)
            img_test = cv2.imread(image_path)
            
            cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB, dst=img_test)
            
            ax1 = fig.add_subplot(rows, cols, count+1)
            ax1.set_title(f'{i}')
            ax1.grid(False)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.imshow(img_test)

            count += 1

            if count % (rows*cols) == 0:
                plt.savefig('./savefig_default.png')
                count = 0
