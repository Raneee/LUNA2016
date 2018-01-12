import os
from random import shuffle
import pickle
import copy
import Tools_Image as IMG_T


import SimpleITK as sitk
import Tools_Augmentation as AUG_T
import copy
import numpy as np



def classFromOutput(output):
    all_class = ['benign','malignant']
    class_list = []
    class_i_list = []
    top_n, top_i = output.data.topk(1)
    
    for idx in range(top_i.size()[0]):
        class_i_list.append(top_i[idx][0])
        class_list.append(all_class[top_i[idx][0]])
        
    return class_list, class_i_list



def makePreLists(index, isBalanced=False, isTest=False, isUndersampling=False):


    candPath = '../Data/CSVFILES/candidates_V2.csv'
    f = file(candPath, 'r')
    lines = f.readlines()
    f.close() 
    
    
    wholeCandDict = {}
    for wholeline in lines[1:]:
        line = wholeline.split(',')

        if line[0] in wholeCandDict.keys():
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            wholeCandDict[line[0]]['List'].append(tempDict)
        else:
            tempList = []
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            tempList.append(tempDict)

            ttempDict = {}
            ttempDict['List'] = tempList
            ttempDict['IMG'] = None
            ttempDict['Origin'] = None
            ttempDict['Spacing'] = None
            wholeCandDict[line[0]] = ttempDict    

            
            
    subset_path = os.path.join('../Data/SUBSET', 'subset'+ str(index))
    p_files = os.listdir(subset_path)
    
    patientDict = {}
    for patient in p_files:
        if '.mhd' in patient: 
            patient = patient.replace('.mhd', '')

            Image, Origin, Spacing = IMG_T.imgReturn(patient)
            patientDict[patient] = copy.deepcopy(wholeCandDict[patient])
            patientDict[patient]['IMG'] = Image
            patientDict[patient]['Origin'] = Origin
            patientDict[patient]['Spacing'] = Spacing

    balancedCandidate = []

    p_cnt = 0
    n_cnt = 0
    for patient in patientDict:
        for Candidate in patientDict[patient]['List']:
            if Candidate['Label'] == '1':
                p_cnt += 1
            else:
                n_cnt += 1
    ratio = n_cnt / p_cnt
    



    if isUndersampling:
        print p_cnt, n_cnt
        balanced_ratio = 30
        patient_list = patientDict.keys()
        shuffle(patient_list)
        t_p_cnt = 0
        for patient in patient_list:
            c_p_cnt = 0
            c_n_cnt = 0
            shuffle(patientDict[patient]['List'])
            for Candidate in patientDict[patient]['List']:
                if t_p_cnt < int(p_cnt * 0.8):
                    if Candidate['Label'] == '1':

                        for i in range(balanced_ratio):
                            infoDict = {}
                            infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                            infoDict['P_ID'] = patient
                            infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                            balancedCandidate.append(infoDict)
                        c_p_cnt += 1
                        t_p_cnt += 1
                print t_p_cnt
        
            for Candidate in patientDict[patient]['List']:
                if c_p_cnt * balanced_ratio > c_n_cnt:
                    if Candidate['Label'] != '1':
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
                        c_n_cnt += 1



    	shuffle(balancedCandidate)
        print 'UNDER SAMPLING --> 80% of Positive, *30 of Negative'
    
    	return patientDict, balancedCandidate


    if not isBalanced:     
        for patient in patientDict:
            for Candidate in patientDict[patient]['List']:
                if isTest:
                    infoDict = {}
                    infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                    infoDict['P_ID'] = patient
                    infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                    balancedCandidate.append(infoDict)
                else:
                    if Candidate['Label'] == '1':
                        for i in range(ratio):
                            infoDict = {}
                            infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                            infoDict['P_ID'] = patient
                            infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                            balancedCandidate.append(infoDict)
                    else:
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
        print 'USING ALL DATA ----> No BALANCING'
    else:
        balanced_ratio = 100
        for patient in patientDict:
            c_p_cnt = 0
            c_n_cnt = 0
            shuffle(patientDict[patient]['List'])
            for Candidate in patientDict[patient]['List']:
                if Candidate['Label'] == '1':

                    for i in range(balanced_ratio):
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
                    c_p_cnt += 1
            for Candidate in patientDict[patient]['List']:
                if c_p_cnt * balanced_ratio > c_n_cnt:
                    if Candidate['Label'] != '1':
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
                        c_n_cnt += 1


        print 'USING BALANCED DATA -----> * 100'
    shuffle(balancedCandidate)
    
    return patientDict, balancedCandidate






def modify_candidates_V2_OUT(p_ID, XYZ, Prob):
    lines = []
    for idx in range(len(p_ID)):

        input_p_id = p_ID[idx]
        input_x = XYZ[idx][0]
        input_y = XYZ[idx][1]
        input_z = XYZ[idx][2]
        input_Prob = Prob[idx][1]
        input_list = [input_p_id, input_x, input_y, input_z, input_Prob]

        line = ','.join(str(e) for e in input_list) + '\n'
        lines.append(line)
    return lines









































































'''


def read_candidates_V2(cand_path):
    candDict = {}
    
    f = file(cand_path, 'r')
    lines = f.readlines()
    f.close()
    
    for wholeline in lines[1:]:
        line = wholeline.split(',')

        if line[0] in candDict.keys():
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            candDict[line[0]]['List'].append(tempDict)
        else:
            tempList = []
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            tempList.append(tempDict)
            ttempDict = {}
            ttempDict['List'] = tempList
            ttempDict['IMG'] = None
            ttempDict['Origin'] = None
            ttempDict['Spacing'] = None
            candDict[line[0]] = ttempDict    
                
    return candDict


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    
    return numpyimage, numpyOrigin, numpySpacing
def imgReturn(p_id):
    img_Dir_Path = '../Data/ALLINONE'

    imgPath = os.path.join(img_Dir_Path, p_id + '.mhd')
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(imgPath)
    
    return numpyImage, numpyOrigin, numpySpacing







def makeCandidateList(patientDict, isBalanced=False, isTest=False):
    
    balancedCandidate = []

    p_cnt = 0
    n_cnt = 0
    for patient in patientDict:
        for Candidate in patientDict[patient]['List']:
            if Candidate['Label'] == '1':
                p_cnt += 1
            else:
                n_cnt += 1
    ratio = n_cnt / p_cnt
    print p_cnt, '!!!!'
    print n_cnt, '!!!!'
    
    if not isBalanced:     
        for patient in patientDict:
            for Candidate in patientDict[patient]['List']:
                if isTest:
                    infoDict = {}
                    infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                    infoDict['P_ID'] = patient
                    infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                    balancedCandidate.append(infoDict)
                else:
                    if Candidate['Label'] == '1':
                        for i in range(ratio):
                            infoDict = {}
                            infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                            infoDict['P_ID'] = patient
                            infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                            balancedCandidate.append(infoDict)
                    else:
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
    else:
        balanced_ratio = 10
        for patient in patientDict:
            c_p_cnt = 0
            c_n_cnt = 0
            shuffle(patientDict[patient]['List'])
            for Candidate in patientDict[patient]['List']:
                if Candidate['Label'] == '1':

                    for i in range(balanced_ratio):
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
                    c_p_cnt += 1
            for Candidate in patientDict[patient]['List']:
                if c_p_cnt * balanced_ratio > c_n_cnt:
                    if Candidate['Label'] != '1':
                        infoDict = {}
                        infoDict['XYZ'] = copy.deepcopy(Candidate['XYZ'])
                        infoDict['P_ID'] = patient
                        infoDict['Label'] = copy.deepcopy(Candidate['Label'])
                        balancedCandidate.append(infoDict)
                        c_n_cnt += 1



    shuffle(balancedCandidate)
    
    return balancedCandidate


def makePatientDict(index):

    candPath = '../Data/CSVFILES/candidates_V2.csv'
    f = file(candPath, 'r')
    lines = f.readlines()
    f.close() 
    
    
    wholeCandDict = {}
    for wholeline in lines[1:]:
        line = wholeline.split(',')

        if line[0] in wholeCandDict.keys():
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            wholeCandDict[line[0]]['List'].append(tempDict)
        else:
            tempList = []
            tempDict = {}
            tempDict['Label'] = line[4][:-2]
            tempDict['XYZ'] = (line[1], line[2], line[3])
            tempList.append(tempDict)

            ttempDict = {}
            ttempDict['List'] = tempList
            ttempDict['IMG'] = None
            ttempDict['Origin'] = None
            ttempDict['Spacing'] = None
            wholeCandDict[line[0]] = ttempDict    

            
            
    subset_path = os.path.join('../Data/SUBSET', 'subset'+ str(index))
    p_files = os.listdir(subset_path)
    
    candDict = {}
    for patient in p_files:
        if '.mhd' in patient: 
            patient = patient.replace('.mhd', '')

            Image, Origin, Spacing = imgReturn(patient)
            candDict[patient] = copy.deepcopy(wholeCandDict[patient])
            candDict[patient]['IMG'] = Image
            candDict[patient]['Origin'] = Origin
            candDict[patient]['Spacing'] = Spacing
    return candDict

'''

