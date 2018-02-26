import os
import model as models
import resnet3D as r3
import densenet3D as d3
import resnet2D as r2
import densenet2D as d2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import math

class ResRes(nn.Module):
    def __init__(self, sample_size):
        super(ResRes, self).__init__()
        Res2D = r2.ResNet(18, sample_size, without_fc=True)
        Res3D = r3.generate_3DResnet('resnet', 18, sample_size, 2, without_fc=True)
        self.sample_size = sample_size

    def forward(self, img2D, img3D):





        out2D = Res2D(img2D)
        print out2D
        out3D = Res3D(img3D)
        print out3D
        concat = torch.cat([out2D, out3D], 1)
        out = nn.Linear(concat, 2)

        return out
#########################modify!!!!
