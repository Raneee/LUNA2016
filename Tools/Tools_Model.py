import os
import sys
sys.path.insert(0, 'Model')
import model as models
import resnet3D as r3
import densenet3D as d3
import resnet2D as r2
import densenet2D as d2
#import addnet as add

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



from collections import OrderedDict


model_names = ['ResNet', '3DNet', '2D3DNet', 'Resnet3D', 'Densenet3D', 'Densenet2D']

def model_setter(idx, img_size=64, batch_size=None, isTest=False):
    if batch_size != None:
        default_batch = batch_size
    else:
        default_batch = 64
    


    '''
    if idx == 0:
        model_name = 'ResNet'
        batch_size = default_batch
        #model = models.CNNfor2D_Small(64)
        model = r2.ResNet(18, img_size)
    elif idx == 1:
        model_name = '3DNet'
        batch_size = default_batch
        #model = models.CNNfor3D(2)
        model = models.CNNfor3D_DIFF(2)
    elif idx == 2:
        model_name = '2D3DNet'
        batch_size = default_batch
        #model = models.CNNfor2D3D(100)
        model = models.CNNfor2D3D_DIFF(img_size)
    elif idx == 3:
        model_name = 'Resnet3D'
        batch_size = default_batch
        model, _ = r3.generate_3DResnet('resnet', 18, img_size, 2, isTest=isTest)
    elif idx == 4:    
        model_name = 'Densenet3D'
        batch_size = default_batch
        model, _ = d3.generate_3DDensenet('densenet', 121, img_size, 2)
    elif idx == 5:
        model_name = 'Densenet2D'
        batch_size = default_batch
        model = d2.DenseNet(121, img_size)
    '''
    if idx == 0:
        model_name = 'Resnet3D'
        batch_size = default_batch
        model, _ = r3.generate_3DResnet('resnet', 18, img_size, 2, isTest=isTest)
    elif idx == 1:
        model_name = 'Densenet3D'
        batch_size = default_batch
        model, _ = d3.generate_3DDensenet('densenet', 121, img_size, 2)
    '''
    elif idx == 2:
        model_name = '3DNet'
        batch_size = default_batch
        #model = models.CNNfor3D(2)
        model = models.CNNfor3D_DIFF(2)
    elif idx == 3:
        model_name = 'Densenet2D'
        batch_size = default_batch
        model = d2.DenseNet(121, img_size)
    elif idx == 4:   
        model_name = 'ResNet'
        batch_size = default_batch
        #model = models.CNNfor2D_Small(64)
        model = r2.ResNet(18, img_size) 
    elif idx == 5:
        model_name = '2D3DNet'
        batch_size = default_batch
        #model = models.CNNfor2D3D(100)
        model = models.CNNfor2D3D_DIFF(img_size)

    '''
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        
    return model, model_name, batch_size



def modelLoader(model_name, test_index, img_size, epoch=-1):
    if not os.path.exists('../Model'):
        os.mkdir('../Model')


    model_path = os.path.join('../Model', model_name + '_withoutPT')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    files = os.listdir(model_path)
    model_list = []
    for file in files:
        if '.pt' in file and (model_name + '____' + str(test_index)) in file:
            if ('__' + str(img_size)) in file: 
                model_list.append(file)

    model_list.sort()
    print model_path
    print model_list
    if len(model_list) < 1:
        return None, -1
    else:
        if epoch != -1:
            model_name = model_name + '____' + str(test_index) + '__' + str(epoch) + '__' + str(img_size) + '.pt'
            if os.path.isfile(os.path.join(model_path, model_name)):
                model_out = os.path.join(model_path, model_name)
                model_epoch = epoch
            else:
                print 'NO MODEL!!'
                model_out = os.path.join(model_path, model_list[-1])
                model_epoch = int(model_list[-1].split('__')[-2])
        else:
            model_out = os.path.join(model_path, model_list[-1])
            model_epoch = int(model_list[-1].split('__')[-2])  


        return model_out, model_epoch


