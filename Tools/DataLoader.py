import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import Tools_Augmentation as AUG_T
import Tools_Image as IMG_T
import numpy as np

class my_dataset_byInfo(data.Dataset):
    def initialize(self, CandidateList, PatientList):
        self.CandidateList = CandidateList
        self.PatientList = PatientList
        self.size = len(CandidateList)

    def __getitem__(self, idx):
        img_tensor_list = []
        
        img_3D, label, P_ID, XYZ = IMG_T.makeCutImage(self.CandidateList[idx], self.PatientList)
        img_3D = AUG_T.randomCrop(img_3D, size=64)
        if label == '1':
            img_3D = AUG_T.rotate_3D(img_3D)
        img3D_50, img3D_75, img3D_100 = converter3D(img_3D)
        
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
    img_64 = img_3D
    img_48 = AUG_T.getSmallImage_3D(img_3D, 48)
    img_32 = AUG_T.getSmallImage_3D(img_3D, 32)
    
    
    
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
    img2D = torch.from_numpy(img2D).float()
        
    return img2D
