import os
import pickle
import torch
from torchvision import transforms
from PIL import Image

from attack import add_gausian_noise

import shutil

def copy_and_paste(source_file_path, destination_folder_path):
    try:
        shutil.copy2(source_file_path, destination_folder_path)
    except IOError as e:
        print("An error occurred while copying a file.: ", str(e))


def write_list_to_file(file_path, my_list):
    try:
        # If the file path doesn't exist, create a directory.
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(file_path, 'w') as file:
            for item in my_list:
                file.write(str(item) + '\n')
    except IOError as e:
        print("An error occurred while writing a file.: ", str(e))


folder_path = 'YOUR_DATASET_PATH(ex: dataset/CelebA/images/)'
attr_path = 'YOUR_ATTR_PATH(ex: dataset/CelebA/images/list_attr_celeba.txt)'

# Split dataset
folder_path_a = './dataset/CelebA/images_a/'    # Assuming the defender already has the data, use it to train defense models.
folder_path_b = './dataset/CelebA/images_b/'    # Training detection models, assuming the defender has collected the data to train detection models.
folder_path_c = './dataset/CelebA/images_c/'    # Used to validate detection model accuracy, assuming the data is input to the actual detection model as it operates.

attr_a = []
attr_b = []
attr_c = []


file_list = os.listdir(folder_path)
jpg_files = [file for file in file_list if file.endswith('.jpg')]
jpg_files = sorted(jpg_files)

lines = [line.rstrip() for line in open(attr_path, 'r')]


for i, file_name in enumerate(jpg_files):
    file_path_ori = os.path.join(folder_path, file_name)

    if i < len(file_list)*0.7:
        file_path_new = os.path.join(folder_path_a, file_name)
        attr_a.append(lines[i])
    elif len(file_list)*0.7 <= i < len(file_list)*0.9:
        file_path_new = os.path.join(folder_path_b, file_name)
        attr_b.append(lines[i])
    else:
        file_path_new = os.path.join(folder_path_c, file_name)
        attr_c.append(lines[i])

    copy_and_paste(file_path_ori, file_path_new)


write_list_to_file(os.path.join(folder_path_a, 'list_attr_celeba.txt'), attr_a)
write_list_to_file(os.path.join(folder_path_b, 'list_attr_celeba.txt'), attr_b)
write_list_to_file(os.path.join(folder_path_c, 'list_attr_celeba.txt'), attr_c)

