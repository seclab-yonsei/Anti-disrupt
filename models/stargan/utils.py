import argparse
import os

# import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

tf_toPILImage = ToPILImage() 


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def save_images(images, epoch=None, path='results', image_num=0):
    att_num = len(images)

    for attr_num in range(att_num):
        for i in range(len(images[attr_num])):
            image = images[attr_num][i]
            img_path = os.path.join(path, '%d'%(image_num+i))
            createFolder(img_path)
            img_path = os.path.join(img_path, '%d.jpg'%(attr_num))
            save_image(image, img_path)
    
    image_num += len(images[0])

    return image_num

def ganimation_save_image(imgtensor, dir, filemame):
    createFolder(dir)
    save_image((imgtensor+1)/2, os.path.join(dir, filemame))
    
def PIL_save_images(images, dir):
    '''
    in ganimation
    '''
    for i in range(len(images)):
        path=os.path.join(dir, '%d-images.jpg'%(i+1))
        images[i].save(path)
        print(path)
    
def str2bool(v):
    return v.lower() in ('true')

'''def get_celebA_sample_image2(data_loader, num, device):
    
    image_list = list()
    for i, (x_real, c_org) in enumerate(data_loader):
        if i < num:
            if i==0: print(x_real.shape)
            # Prepare input images and target domain labels.
            x_real = x_real.to(device)
            image_list.append(x_real)
    return image_list

def get_celebA_sample_image(data_loader, num, device):
    image_list = list()
    for i, (x_real, c_org) in enumerate(data_loader):
        if i < num:
            if i==0: print(x_real.shape)
            # Prepare input images and target domain labels.
            x_real = x_real.to(device)
            image_list.append(x_real)
    return image_list
'''
def get_concat_h(imglist):
    '''
    tensor img list -> 1 Pillow img
    '''
    dst = Image.new('RGB', (imglist[0].width *len(imglist), imglist[0].height))
    for i in range(len(imglist)):
        dst.paste(imglist[i], (i*imglist[0].width, 0))
    return dst


def ganimation_get_concat_h(img_t_list):
    # print((img_t_list[0].size()))
    # print((((img_t_list[0]+1)/2).size()))
    img = tf_toPILImage((img_t_list[0][0]+1)/2)

    dst = Image.new('RGB', (img.width *len(img_t_list), img.height))
    for i in range(len(img_t_list)):
        for j in range(len(img_t_list[0])):
            img=tf_toPILImage((img_t_list[i][j]+1)/2)
            dst.paste(img, (i*img.width, 0))
    return dst

def print_config(name: str, config):
    print('--------------------',name,' configration--------------------')
    print(config)
    print('\n\n\n') 

def save_config_dict(config_dict: dict, file_path: str):
    list_keys = list(config_dict.keys())
    with open(file_path, "w") as f:
        for key in list_keys:
            temp_str = key + ": " + str(config_dict[key]) + '\n'
            f.write(temp_str)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')