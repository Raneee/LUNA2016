import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import math
from functools import partial



class DenseNet(nn.Module):
    def __init__(self, model_depth, sample_size, pretrained=True, num_classes=2, without_fc=False):
        super(DenseNet, self).__init__()
        if model_depth == 121:
            CNN_model = models.densenet121(pretrained=pretrained)
        elif model_depth == 161:
            CNN_model = models.densenet161(pretrained=pretrained)
        elif model_depth == 169:
            CNN_model = models.densenet169(pretrained=pretrained)
        elif model_depth == 201:
            CNN_model = models.densenet201(pretrained=pretrained)

        self.num_ftrs = CNN_model.classifier.in_features
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
        out = self.fc(fin_feature)
        return out
        if self.without_fc:
            return out
        else:
            out = self.fc(fin_feature)
            return out

