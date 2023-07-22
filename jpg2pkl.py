import os
import pickle
import torch
from torchvision import transforms
from PIL import Image

from attack import add_gausian_noise

# 이미지가 저장된 폴더 경로
folder_path = './dataset/CelebA/images_c/'

# 변환을 위한 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 폴더 내의 모든 파일 목록 가져오기
file_list = os.listdir(folder_path)

# jpg 파일만 필터링하여 리스트에 추가
jpg_files = [file for file in file_list if file.endswith('.jpg')]
jpg_files = sorted(jpg_files)

# 이미지 데이터를 저장할 리스트 생성
image_data = []
adv_data = []

# 각 이미지 파일을 열어서 텐서로 변환하여 리스트에 저장
count = 0
for file_name in jpg_files:
    file_path = os.path.join(folder_path, file_name)
    image = preprocess(Image.open(file_path))
    image_data.append(image)
    # x_adv = add_gausian_noise(image.cuda(), 0.004)
    # adv_data.append(x_adv.cpu())

# 이미지 데이터를 pickle 파일로 저장
pickle_file_path = './dataset/CelebA/pickle/original_c.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(image_data, pickle_file)

# pickle_file_path = './dataset/CelebA/pickle/adv_original.pkl'
# with open(pickle_file_path, 'wb') as pickle_file:
#     pickle.dump(adv_data, pickle_file)
