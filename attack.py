import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

def add_gausian_noise(original_image, var=1.0, device='cuda:0'):
    noise_image = original_image + torch.normal(0, var, size=original_image.shape).to(device)
    noise_image = torch.clamp(noise_image, -1.0, 1.0)
    return noise_image

## TODO
## FGSM, PGD, MIM 공격수행하는 python 패키지는 없나?
## 없으면 걍 폴더에서 각각...하돈가...