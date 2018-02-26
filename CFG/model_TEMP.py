import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import math


class CNNfor2D_Small(nn.Module):
    def __init__(self, sample_size, num_classes=2, avg_kern=2):
        super(CNNfor2D_Small, self).__init__()
        CNN_model = models.resnet18(pretrained=True)
        self.num_ftrs = CNN_model.fc.in_features
        self.features = list(CNN_model.children())
        for i in range(2):
            self.features.pop()
        last_size = int(math.ceil(sample_size / 32.))
        self.features.append(nn.AvgPool2d(last_size))
        self.CNN_model = nn.Sequential(*self.features)
        self.fc = nn.Linear(self.num_ftrs, num_classes)
        
        
    def forward(self, images):
        out = self.CNN_model(images)
        fin_feature = out.view(images.size(0), -1)
        out = self.fc(fin_feature)
        return out





class CNNfor3D_DIFF(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNfor3D_DIFF, self).__init__()
        
        self.fc = nn.Linear(32 * 3, num_classes)
        
        self.model_64 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(13, 13, 13))
        )
        
        self.model_48 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(9, 9, 9))
        )
        
        self.model_32 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(5, 5, 5))
        )
        
        
    def forward(self, x, y, z):
        out_1 = self.model_32(x)
        out_2 = self.model_48(y)
        out_3 = self.model_64(z)
        out_1 = out_1.view(-1, 32)
        out_2 = out_2.view(-1, 32)
        out_3 = out_3.view(-1, 32)
        
        out = torch.cat([out_1, out_2, out_3], 1)
        out = self.fc(out)
        
        return out


    
class CNNfor2D3D_DIFF(nn.Module):
    def __init__(self, sample_size, num_classes=2, avg_kern=2):
        super(CNNfor2D3D_DIFF, self).__init__()
        
        
        CNN2D_model = models.resnet18(pretrained=True)
        self.num_ftrs = CNN2D_model.fc.in_features
        self.features = list(CNN2D_model.children())
        for i in range(2):
            self.features.pop()
        last_size = int(math.ceil(sample_size / 32.))
        self.features.append(nn.AvgPool2d(last_size))
        self.CNN2D_model = nn.Sequential(*self.features)
        self.fc = nn.Linear(self.num_ftrs, 32)
        
        
        
        self.fc_final = nn.Linear(32 * 4, num_classes)
        
        self.model_64 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(13, 13, 13))
        )
        self.model_48 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(9, 9, 9))
        )
        self.model_32 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(6, 16, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(16, 32, kernel_size=(2, 2, 2)),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(5, 5, 5))
        )
        
        
        
    def forward(self, x, y, z, s):
        out_1 = self.model_32(x)
        out_1 = out_1.view(-1, 32)
        out_2 = self.model_48(y)
        out_2 = out_2.view(-1, 32)
        out_3 = self.model_64(z)
        out_3 = out_3.view(-1, 32)
        
        
        out2D = self.CNN2D_model(s)
        fin_feature = out2D.view(s.size(0), -1)
        out2D = self.fc(fin_feature)
        
        out = torch.cat([out_1, out_2, out_3, out2D], 1)
        out = self.fc_final(out)
        
        
        return out
    














