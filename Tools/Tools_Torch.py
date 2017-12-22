import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def imageOnTorch(img, model_idx, img_size=64):
    if model_idx == 0 or model_idx == 5:
        img_32 = to_var(torch.from_numpy(img[3]).float())
        return img_32
    elif model_idx == 1:
        img_32 = to_var(torch.from_numpy(img[0]).float())
        img_48 = to_var(torch.from_numpy(img[1]).float())
        img_64 = to_var(torch.from_numpy(img[2]).float())
        return img_32, img_48, img_64
    elif model_idx == 2:
        
        img_32 = to_var(torch.from_numpy(img[0]).float())
        img_48 = to_var(torch.from_numpy(img[1]).float())
        img_64 = to_var(torch.from_numpy(img[2]).float())
        img_2D = to_var(torch.from_numpy(img[3]).float())
        return img_32, img_48, img_64, img_2D
    elif model_idx == 6:
        img_64 = to_var(torch.from_numpy(img[2]).float())
        img_2D = to_var(torch.from_numpy(img[3]).float())
        return img_2D, img_64
    else:
        if img_size == 32:
            convert_img = img[0]
        elif img_size == 64:
            convert_img = img[2]
        else:
            convert_img = img[1]
        if convert_img.shape[1] == 1:
            convert_img = np.concatenate((convert_img, convert_img, convert_img), axis = 1) 
            convert_img = to_var(torch.from_numpy(convert_img).float())
        return convert_img

