import sys
sys.path.insert(0, 'Tools')
sys.path.insert(0, 'CFG')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
import os
import model as MODEL
import DataLoader as DL
import Tools_Torch as TORCH_T
import Tools_IO as IO_T

from PIL import Image


def train(idx, batch_size):
    cand_path = '../Data/CSVFILES/candidates_V2.csv'
    candidate_V2 = IO_T.read_candidates_V2(cand_path)

    for epoch in range(1):
        for test_index in range(1):
            for train_index in range(2):
                if train_index != test_index:

                    train_correct_cnt = 0
                    print '      Train for ', train_index + 1, ' fold'
                    
                    patientDict = IO_T.makePatientDict(train_index, candidate_V2)
                    balancedCandidate = IO_T.makeBalancedList(patientDict)
                    
                    print '          Patient Count : ', len(patientDict)
                    print '          Nodule Count : ', len(balancedCandidate)
                    for p in patientDict:
                        print p
                        print patientDict[p].IMG.shape
                    
                    dataset = DL.my_dataset_byInfo()
                    dataset.initialize(balancedCandidate, patientDict)
                    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
                    for batch_index, (img_tensor, label_tensor, P_ID, XYZ) in enumerate(train_loader):
                        img_32 = TORCH_T.to_var(img_tensor[0])
                        img_48 = TORCH_T.to_var(img_tensor[1])
                        img_64 = TORCH_T.to_var(img_tensor[2])
                        img_2D = TORCH_T.to_var(img_tensor[3])
                        label = TORCH_T.to_var(label_tensor.view(-1))
                        
                        
                        img = transforms.ToPILImage()(img_tensor[3][0])
                        print img.shape
                        img.save('out.png')
                        
                        break
                    
train(0, 1)