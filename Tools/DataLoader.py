import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import Augmentation as AUG
from scipy.misc import imread, imresize
import Tools_Image as IMG_T
import numpy as np

class my_dataset_byPath(data.Dataset):
    def initialize(self, nodule_list):
        self.nodule_list = nodule_list
        self.size = len(nodule_list)

    def __getitem__(self, idx):
        img_tensor_list = []
        path = self.nodule_list[idx]['path']
        label = self.nodule_list[idx]['type']
            
        img_3D = np.load(path)
        if img_3D.shape[0] != 64 or img_3D.shape[1] != 64 or img_3D.shape[2] != 64:
            img_3D = AUG.makeSquare(img_3D, 64)
        img_3D = AUG.randomCrop(img_3D)
        
        if label == '1':
            img_3D = AUG.rotate_3D(img_3D)
        img3D_100, img3D_75, img3D_50 = converter3D(img_3D)
        img2D = converter2D(img_3D)
        
        img_tensor_list.append(img3D_100)
        img_tensor_list.append(img3D_75)
        img_tensor_list.append(img3D_50)
        img_tensor_list.append(img2D)
        
        
        label_tensor = torch.LongTensor([int(label)])
        
        return img_tensor_list, label_tensor
    
    def __len__(self):
        return self.size
    
    
    

    

class my_dataset_byInfo(data.Dataset):
    def initialize(self, CandidateList, PatientList):
        self.CandidateList = CandidateList
        self.PatientList = PatientList
        self.size = len(CandidateList)

    def __getitem__(self, idx):
        img_tensor_list = []
        
        img_3D, label, P_ID, XYZ = IMG_T.makeCutImage(self.CandidateList[idx], self.PatientList)
        img_3D = AUG.randomCrop(img_3D, size=64)
        if label == '1':
            img_3D = AUG.rotate_3D(img_3D)
        img3D_50, img3D_75, img3D_100 = converter3D_NEW(img_3D)
        
        img2D = converter2D(img_3D)
        img_tensor_list.append(img3D_50)
        img_tensor_list.append(img3D_75)
        img_tensor_list.append(img3D_100)
        img_tensor_list.append(img2D)
        
        
        label_tensor = torch.LongTensor([int(label)])
        
        return img_tensor_list, label_tensor, P_ID, XYZ
    
    def __len__(self):
        return self.size

    
def converter3D(img_3D):
    img3D_100 = torch.from_numpy(img_3D).float()
    img3D_100 = img3D_100.view(1, 64, 64, 64)
            
    zoom_img = AUG.zoomIn_3D(img_3D, 0.75)
    img3D_75 = torch.from_numpy(zoom_img).float()
    img3D_75 = img3D_75.view(1, 64, 64, 64)
        
        
    zoom_img = AUG.zoomIn_3D(img_3D, 0.5)
    img3D_50 = torch.from_numpy(zoom_img).float()
    img3D_50 = img3D_50.view(1, 64, 64, 64)
        
    return img3D_50, img3D_75, img3D_100

def converter3D_NEW(img_3D):
    img_64 = img_3D
    img_48 = AUG.getSmallImage_3D(img_3D, 48)
    img_32 = AUG.getSmallImage_3D(img_3D, 32)
    
    
    
    img3D_100 = torch.from_numpy(img_64).float()
    img3D_100 = img3D_100.view(1, 64, 64, 64)
            
        
    img3D_75 = torch.from_numpy(img_48).float()
    img3D_75 = img3D_75.view(1, 48, 48, 48)
        
        
    img3D_50 = torch.from_numpy(img_32).float()
    img3D_50 = img3D_50.view(1, 32, 32, 32)
        
    return img3D_50, img3D_75, img3D_100
    
def converter2D(img_3D, size2D=64):
    img2D_X = img_3D[32, :, :]
    img2D_Y = img_3D[:, 32, :]
    img2D_Z = img_3D[:, :, 32]
        
    img2D_X = np.expand_dims(img2D_X, axis=0)
    img2D_Y = np.expand_dims(img2D_Y, axis=0)
    img2D_Z = np.expand_dims(img2D_Z, axis=0)
    img2D = np.concatenate((img2D_X, img2D_Y, img2D_Z), axis = 0) 
    #img2D = imresize(img2D, (size2D, size2D))
    img2D = torch.from_numpy(img2D).float()
        
    return img2D