import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


# mutation 함수 선택
def choose_fun(img, choose, params):
    if choose == 0:
        return image_translation(img, params)  # -3, 3
    elif choose == 1:
        return image_scale(img, params * 0.1)  # 7, 12
    elif choose == 2:
        return image_shear(img, params * 0.1)  # -6, 6
    elif choose == 3:
        return image_contrast(img, params * 0.1)  # 5, 13
    elif choose == 4:
        return image_rotation(img, params)  # -50, 50
    elif choose == 5:
        return image_brightness(img, params)  # -20, 20
    elif 6 <= choose <= 9:
        return image_blur(img, params, choose)  # 1, 10
    elif choose == 10:
        return image_pixel_change(img, params)  # 1, 10
    elif choose == 11 or choose == 12:
        return image_noise(img, params, choose)  # 1, 4


# 이미지 위치 변화
def image_translation(img, params):
    rows, cols, ch = img.shape
    # rows, cols = img.shape
    # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    M = np.float32([[1, 0, params], [0, 1, params]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# 크기 축소
def image_scale(img, params):
    # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    rows, cols, ch = img.shape
    res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
    res = res.reshape((res.shape[0], res.shape[1], ch))
    y, x, z = res.shape
    if params > 1:  # need to crop
        startx = x // 2 - cols // 2
        starty = y // 2 - rows // 2
        return res[starty:starty + rows, startx:startx + cols]
    elif params < 1:  # need to pad
        sty = round((rows - y) / 2)
        stx = round((cols - x) / 2)
        return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                      constant_values=0)
    return res


# x와 y를 기준으로 밀림변환, 찌그러짐
def image_shear(img, params):
    rows, cols, ch = img.shape
    # rows, cols = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# 이미지 회전
def image_rotation(img, params):
    rows, cols, ch = img.shape
    # rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
    return dst


# 이미지 대조
def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
    # new_img = cv2.add(mul_img, beta)              # new_img = img*alpha + beta
    # new_img[0][0] = 1
    # print(img - new_img)

    return new_img


# 이미지 밝기
def image_brightness(img, params):
    beta = params   # 배열 새로 생성하여 더하는 방법 시ㄷ
    new_img = img + beta
    new_img = np.clip(new_img, 0, 255)
    # new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
    return new_img


# 이미지 흐리게
def image_blur(img, params, temp):
    # print("blur")

    if params == 0:
        return img
    elif temp == 6:
        blur = cv2.blur(img, (params, params))
    elif temp == 7:
        blur = cv2.GaussianBlur(img, (params, params), 0)
    elif temp == 8:
        blur = cv2.medianBlur(img, params)
    elif temp == 9:
        if params == 6:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
        elif params == 9:
            blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


# 이미지 픽셀 변환
def image_pixel_change(img, params):
    # random change 1 - 5 pixels from 0 -255
    img_shape = img.shape
    img1d = np.ravel(np.copy(img))
    arr = np.random.randint(0, len(img1d), params)
    for i in arr:
        img1d[i] = np.random.randint(0, 256)
    new_img = img1d.reshape(img_shape)
    return new_img


# 이미지 노이즈
def image_noise(img, params, temp):
    if temp == 11:  # Gaussian-distributed additive noise.
        row, col, ch = img.shape
        mean = 0
        var = params  # 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)
    elif temp == 12:
        s_vs_p = 0.5
        amount = params  # 0.004
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 255
        # out[tuple(coords)] = 0

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        out = np.clip(out, 0, 255)
        return out
    elif temp == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = img + img * gauss
        return noisy.astype(np.uint8)

