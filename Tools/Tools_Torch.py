import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import model as models

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def model_setter(idx, isTest=False):
    if idx == 0:
        model_name = 'ResNet'
        batch_size = 128
        if isTest:
            return model_name, batch_size
        model = models.CNNfor2D_Small(2)
        if torch.cuda.is_available():
            model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        
        
        return model, model_name, batch_size
    elif idx == 1:
        model_name = '3DNet'
        batch_size = 128
        if isTest:
            return model_name, batch_size
        #model = models.CNNfor3D(2)
        model = models.CNNfor3D_DIFF(2)
        if torch.cuda.is_available():
            model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))        
        
        return model, model_name, batch_size
    else:
        model_name = '2D3DNet'
        batch_size = 128
        if isTest:
            return model_name, batch_size
        #model = models.CNNfor2D3D(100)
        model = models.CNNfor2D3D_DIFF()
        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        
        return model, model_name, batch_size
