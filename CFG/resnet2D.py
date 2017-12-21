import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import math
from functools import partial



class ResNet(nn.Module):
    def __init__(self, model_depth, sample_size, pretrained=True, num_classes=2, without_fc=False):
        super(ResNet, self).__init__()
        if model_depth == 18:
            CNN_model = models.resnet18(pretrained=pretrained)
        if model_depth == 34:
            CNN_model = models.resnet34(pretrained=pretrained)
        if model_depth == 50:
            CNN_model = models.resnet50(pretrained=pretrained)
        if model_depth == 101:
            CNN_model = models.resnet101(pretrained=pretrained)
        if model_depth == 152:
            CNN_model = models.resnet152(pretrained=pretrained)

        self.num_ftrs = CNN_model.fc.in_features
        self.features = list(CNN_model.children())
        for i in range(2):
            self.features.pop()
        last_size = int(math.ceil(sample_size / 32.))
        self.features.append(nn.AvgPool2d(last_size))
        self.CNN_model = nn.Sequential(*self.features)
        self.fc = nn.Linear(self.num_ftrs, num_classes)
        self.without_fc = without_fc
        
    def forward(self, images):
        out = self.CNN_model(images) 
        fin_feature = out.view(images.size(0), -1)
        if self.without_fc:
            return fin_feature
        else:
            out = self.fc(fin_feature)
            return out
