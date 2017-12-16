import numpy as np
import os
import SimpleITK as sitk
import Tools_Augmentation as AUG_T
import copy

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return numpyimage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchdVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchdVoxelCoord / spacing
    return voxelCoord

def voxelToWorldCoord(voxelCoord, origin, spacing):
    stretchdVoxelCoord = voxelCoord * spacing
    worldCoord = stretchdVoxelCoord + origin
    return worldCoord

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.0
    
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1
    npzarray[npzarray < 0] = 0
    
    return npzarray

def Image3DOut(voxelCoord, numpyImage, voxelWidth = 70):
    voxelCoord = voxelCoord.astype(int)
    minVoxel_X = max(voxelCoord[1]-voxelWidth/2, 0)
    maxVoxel_X = min(voxelCoord[1]+voxelWidth/2, 512)    
    minVoxel_Y = max(voxelCoord[2]-voxelWidth/2, 0)
    maxVoxel_Y = min(voxelCoord[2]+voxelWidth/2, 512)
    minVoxel_Z = max(voxelCoord[0]-voxelWidth/2, 0)
    maxVoxel_Z = min(voxelCoord[0]+voxelWidth/2, numpyImage.shape[0] - 1)       
    
    
    patch = numpyImage[minVoxel_Z:maxVoxel_Z,minVoxel_X:maxVoxel_X, minVoxel_Y:maxVoxel_Y]
    patch = normalizePlanes(patch)
    
    if patch.shape[0] != voxelWidth or patch.shape[1] != voxelWidth or patch.shape[2] != voxelWidth:
            patch = AUG_T.makeSquare(patch, voxelWidth)    
    return patch

def imgReturn(p_id):
    img_Dir_Path = '../Data/ALLINONE'

    imgPath = os.path.join(img_Dir_Path, p_id + '.mhd')
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(imgPath)
    
    return numpyImage, numpyOrigin, numpySpacing

def makeCutImage(candidateInfo, patientDict): 
    candidate = candidateInfo
    XYZ = candidate['XYZ']
    P_ID = candidate['P_ID']
    Label = candidate['Label']

    patient = patientDict[P_ID]
    Img = patient['IMG']
    Origin = patient['Origin']
    Spacing = patient['Spacing']
        
    worldCoord = np.asarray([float(XYZ[2]), float(XYZ[1]), float(XYZ[0])])
    voxelCoord = worldToVoxelCoord(worldCoord, Origin, Spacing)
    cutIMG = Image3DOut(voxelCoord, Img)
    
    return cutIMG, Label, P_ID, XYZ
