import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import Tools_Augmentation as AUG_T
import Tools_Image as IMG_T
import Tools_IO as IO_T
import numpy as np
import copy



def makeBatch(start, batch_size, CandidateList, PatientDict):
    start_idx = start * batch_size
    img_50_list = [] 
    img_75_list = []
    img_100_list = []
    img_2D_list = []
    batch_label_list = []
    batch_P_ID_list = []
    batch_XYZ_list = []



    if start_idx + batch_size > len(CandidateList):
        batch_size = len(CandidateList) - start_idx


    for idx in range(batch_size):

        idx = idx + start_idx
        img_3D, label, P_ID, XYZ = IMG_T.makeCutImage(CandidateList[idx], PatientDict)

        img_3D = AUG_T.randomCrop(img_3D, size=64)
        if label == '1':
            img_3D = AUG_T.rotate_3D(img_3D)
        img3D_50, img3D_75, img3D_100 = converter3D(img_3D)

        img2D = converter2D(img_3D)

        img_50_list.append(img3D_50)
        img_75_list.append(img3D_75)
        img_100_list.append(img3D_100)
        img_2D_list.append(img2D)
        batch_label_list.append(int(label))
        batch_P_ID_list.append(P_ID)
        batch_XYZ_list.append(XYZ)

    batch_50 = np.asarray(img_50_list)
    batch_75 = np.asarray(img_75_list)
    batch_100 = np.asarray(img_100_list)
    batch_2D = np.asarray(img_2D_list)




    batch_img = [batch_50, batch_75, batch_100, batch_2D]
    batch_label = np.asarray(batch_label_list)
    batch_P_ID = np.asarray(batch_P_ID_list) 
    batch_XYZ = np.asarray(batch_XYZ_list) 


    return batch_img, batch_label, batch_P_ID, batch_XYZ



def converter3D(img_3D):
    img_64 = img_3D
    img_48 = AUG_T.getSmallImage_3D(img_64, 48)
    img_32 = AUG_T.getSmallImage_3D(img_64, 32)
    
    
    img3D_100 = img_64.reshape((1, 64, 64, 64))
    img3D_75 = img_48.reshape((1, 48, 48, 48))
    img3D_50 = img_32.reshape((1, 32, 32, 32))


    return img3D_50, img3D_75, img3D_100
    
def converter2D(img_3D, size2D=64):
    img2D_X = img_3D[32, :, :]
    img2D_Y = img_3D[:, 32, :]
    img2D_Z = img_3D[:, :, 32]

    img2D_X = np.expand_dims(img2D_X, axis=0)
    img2D_Y = np.expand_dims(img2D_Y, axis=0)
    img2D_Z = np.expand_dims(img2D_Z, axis=0)
    img2D = np.concatenate((img2D_X, img2D_Y, img2D_Z), axis = 0) 
        
    return img2D    

