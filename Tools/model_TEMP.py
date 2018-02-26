import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class CNNfor2D(nn.Module):
    def __init__(self, out_size):
        super(CNNfor2D, self).__init__()
        CNN_model = models.resnet18(pretrained=True)
        modules = list(CNN_model.children())[:-1]    # delete the last fc layer.
        self.CNN_model = nn.Sequential(*modules)
        self.linear = nn.Linear(CNN_model.fc.in_features, out_size)
        self.bn = nn.BatchNorm1d(out_size, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        out = self.CNN_model(images)
        out = Variable(out.data)
        out = out.view(out.size(0), -1)
        out = self.bn(self.linear(out))
        
        return out
    
class CNNfor2D_Small(nn.Module):
    def __init__(self, num_classes=2, avg_kern=2):
        super(CNNfor2D_Small, self).__init__()
        CNN_model = models.resnet18(pretrained=True)
        self.num_ftrs = CNN_model.fc.in_features
        self.features = list(CNN_model.children())
        for i in range(2):
            self.features.pop()
        self.features.append(torch.nn.AvgPool2d(avg_kern))
        self.CNN_model = nn.Sequential(*self.features)
        self.fc = nn.Linear(self.num_ftrs, num_classes)
        
        
    def forward(self, images):
        out = self.CNN_model(images)
        fin_feature = out.view(images.size(0), -1)
        #out = Variable(out.data)
        #out = out.view(out.size(0), -1)
        out = self.fc(fin_feature)
        return out
    
class CNNfor2D_Pure(nn.Module):
    def __init__(self, out_size):
        super(CNNfor2D, self).__init__()
        CNN_model = models.resnet18(pretrained=True)
        num_ftrs = CNN_model.fc.in_features
        CNN_model.fc = nn.Linear(num_ftrs, 2)
        
    def forward(self, images):
        out = self.CNN_model(images)
      
        return out    
    
    
class CNNfor3D(nn.Module):
    def __init__(self, out_size):
        super(CNNfor3D, self).__init__()
        
        self.fc1 = nn.Linear(16 * 5 * 5 * 5 * 3, 512)
        self.fc2 = nn.Linear(512, out_size)
        
        self.model_1 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        self.model_2 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        self.model_3 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        
        
        
    def forward(self, x, y, z):
        out_1 = self.model_1(x)
        out_1 = out_1.view(-1, 16 * 5 * 5 * 5)
        out_2 = self.model_2(y)
        out_2 = out_2.view(-1, 16 * 5 * 5 * 5)
        out_3 = self.model_3(z)
        out_3 = out_3.view(-1, 16 * 5 * 5 * 5)
        
        out = torch.cat([out_1, out_2, out_3], 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
  

    
class CNNfor2D3D(nn.Module):
    def __init__(self, out_size, num_classes=2, avg_kern=2):
        super(CNNfor2D3D, self).__init__()
        
        
        CNN2D_model = models.resnet18(pretrained=True)
        self.num_ftrs = CNN2D_model.fc.in_features
        self.features = list(CNN2D_model.children())
        for i in range(2):
            self.features.pop()
        self.features.append(torch.nn.AvgPool2d(avg_kern))
        self.CNN2D_model = nn.Sequential(*self.features)
        self.fc = nn.Linear(self.num_ftrs, out_size)
        
        
        
        self.fc1 = nn.Linear(16 * 5 * 5 * 5 * 3, 512)
        self.fc2 = nn.Linear(512, out_size)
        
        
        self.fc_final = nn.Linear(out_size * 2, num_classes)
        
        self.model_1 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        self.model_2 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        self.model_3 = nn.Sequential(
            nn.Conv3d(1, 6, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3),
            nn.Conv3d(6, 16, kernel_size=(5, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d(3, 3)
        )
        
        
        
    def forward(self, x, y, z, s):
        out_1 = self.model_1(x)
        out_1 = out_1.view(-1, 16 * 5 * 5 * 5)
        out_2 = self.model_2(y)
        out_2 = out_2.view(-1, 16 * 5 * 5 * 5)
        out_3 = self.model_3(z)
        out_3 = out_3.view(-1, 16 * 5 * 5 * 5)
        
        out2D = torch.cat([out_1, out_2, out_3], 1)
        out2D = F.relu(self.fc1(out2D))
        out2D = self.fc2(out2D)
        
        
        out3D = self.CNN2D_model(s)
        fin_feature = out3D.view(s.size(0), -1)
        out3D = self.fc(fin_feature)
        
        out = torch.cat([out2D, out3D], 1)
        out = self.fc_final(out)
        
        
        return out
    
    
class FCforMerge(nn.Module):
    def __init__(self, input_size):
        super(FCforMerge, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(500, 2)
    
    def forward(self, vec2D, vec3D):
        concat = torch.cat([vec2D, vec3D], 1)
        out = F.relu(self.fc1(concat))
        out = self.fc2(out)
        
        return out