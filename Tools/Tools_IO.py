import os
from random import shuffle
import pickle
import copy
import Tools_Image as IMG_T
import dataStructures as DD


def read_candidates_V2(cand_path):
    candDict = {}
    
    f = file(cand_path, 'r')
    lines = f.readlines()
    f.close()
    
    for wholeline in lines[1:]:
        line = wholeline.split(',')
        if not line[0] in candDict.keys():
            new_patientInfo = DD.Patient()
            new_patientInfo.setCandidateList((line[1], line[2], line[3]), line[4])
            
            candDict[line[0]] = new_patientInfo
        else:
            new_patientInfo.setCandidateList((line[1], line[2], line[3]), line[4])
                
                
    return candDict

def classFromOutput(output):
    all_class = ['benign','malignant']
    class_list = []
    class_i_list = []
    top_n, top_i = output.data.topk(1)
    
    for idx in range(top_i.size()[0]):
        class_i_list.append(top_i[idx][0])
        class_list.append(all_class[top_i[idx][0]])
        
    return class_list, class_i_list

def fileLoader(idx, isTrain=True):
    
    if isTrain:
        f = open(os.path.join('/media/hwejin/SSD_1/Code/Github/LUNA2016/DataManage/CV', str(idx), 'train.txt'))
    else:
        f = open(os.path.join('/media/hwejin/SSD_1/Code/Github/LUNA2016/DataManage/CV', str(idx), 'val.txt'))
        
    train_list = pickle.load(f)
    f.close()
    
    return train_list

def makeNoduleList_train(root_path, list, isBalanced = False):
    true_cnt = 0
    false_cnt = 0

    for patient in list:
        type_path = os.path.join(root_path, patient)
        if os.path.isdir(type_path):
            type_list = os.listdir(type_path)
            for type in type_list:
                nodule_path = os.path.join(type_path, type)
                nodule_list = os.listdir(nodule_path)

                for nodule in nodule_list:
                    img_path = os.path.join(nodule_path, nodule, '3D.npy')
                    nodule_dict = {'type' : type, 'path' : img_path}

                    if type == '1':
                        true_cnt += 1
                    else:
                        false_cnt += 1


    augment = false_cnt / true_cnt

    if not isBalanced :

        nodules = []
        for patient in list:
            type_path = os.path.join(root_path, patient)

            if os.path.isdir(type_path):
                type_list = os.listdir(type_path)
                for type in type_list:
                    nodule_path = os.path.join(type_path, type)
                    nodule_list = os.listdir(nodule_path)

                    for nodule in nodule_list:
                        img_path = os.path.join(nodule_path, nodule, '3D.npy')
                        nodule_dict = {'type' : type, 'path' : img_path}

                        if type == '1':
                            for i in range(augment):
                                nodules.append(nodule_dict)
                        else:
                            nodules.append(nodule_dict)
            else:
                pass
    else:
        nodules = []
        true_cnt = 0
        false_cnt = 0
        for patient in list:
            type_path = os.path.join(root_path, patient)

            if os.path.isdir(type_path):
                type_list = os.listdir(type_path)
                for type in type_list:
                    nodule_path = os.path.join(type_path, type)
                    nodule_list = os.listdir(nodule_path)

                    for nodule in nodule_list:
                        img_path = os.path.join(nodule_path, nodule, '3D.npy')
                        nodule_dict = {'type': type, 'path': img_path}

                        if type == '1':
                            nodules.append(nodule_dict)
                            true_cnt += 1
                        else:
                            if true_cnt > false_cnt:
                                nodules.append(nodule_dict)
                                false_cnt += 1
            else:
                pass
    shuffle(nodules)
    return nodules

def makeNoduleList_test(root_path, list):

    nodules = []
    for patient in list:
        type_path = os.path.join(root_path, patient)

        if os.path.isdir(type_path):
            type_list = os.listdir(type_path)
            for type in type_list:
                nodule_path = os.path.join(type_path, type)
                nodule_list = os.listdir(nodule_path)

                for nodule in nodule_list:
                    img_path = os.path.join(nodule_path, nodule, '3D.npy')
                    nodule_dict = {'type' : type, 'path' : img_path}

                    if type == '1':
                        nodules.append(nodule_dict)
                    else:
                        nodules.append(nodule_dict)
        else:
            pass
    shuffle(nodules)
    return nodules

def makeBalancedList(patientDict, isTest=False):
    balancedCandidate = []

    p_cnt = 0
    n_cnt = 0
    for patient in patientDict:
        for Candidate in patientDict[patient].CandidateList:
            if Candidate['Label'] == '1':
                p_cnt += 1
            else:
                n_cnt += 1
    ratio = n_cnt / p_cnt

        
    for patient in patientDict:
        for Candidate in patientDict[patient].CandidateList:
            infoDict = {}
            infoDict['XYZ'] = Candidate['XYZ']
            infoDict['P_ID'] = patient
            infoDict['Label'] = Candidate['Label']
            if isTest:
                balancedCandidate.append(infoDict)
            else:
                if infoDict['Label'] == '1':
                    for i in range(ratio):
                        balancedCandidate.append(infoDict)
                else:
                    balancedCandidate.append(infoDict)

    shuffle(balancedCandidate)
    
    return balancedCandidate


def makePatientDict(index, wholeCandidateList):
    
    subset_path = os.path.join('../Data/SUBSET', 'subset'+ str(index))
    p_files = os.listdir(subset_path)
    
    candDict = {}
    
    for patient in p_files:
        if '.mhd' in patient: 
            patient = patient.replace('.mhd', '')
            patientInfo = copy.deepcopy(wholeCandidateList[patient])
            
            Image, Origin, Spacing = IMG_T.imgReturn(patient)
            patientInfo.setIMG(Image)
            patientInfo.setOrigin(Origin)
            patientInfo.setSpacing(Spacing)
            
            candDict[patient] = patientInfo
                
    return candDict

def modify_candidates_V2_OUT(p_ID, XYZ, Prob):
    lines = []
    for idx in range(len(p_ID)):

        input_p_id = p_ID[idx]
        input_x = XYZ[0][idx]
        input_y = XYZ[1][idx]
        input_z = XYZ[2][idx]
        input_Prob = Prob[idx][1]
        input_list = [input_p_id, input_x, input_y, input_z, input_Prob]

        line = ','.join(str(e) for e in input_list) + '\n'
        lines.append(line)
    return lines

def result_Summary(guess_i, label):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for idx in range(len(guess_i)):
        if str(label[idx]) == '1':
            if str(guess_i[idx]) == '1':
                TP += 1
            else:
                FN += 1
        else:
            if str(guess_i[idx]) == '1':
                FP += 1
            else:
                TN += 1
    return TP, FP, FN, TN




def modelLoader(model_name, test_index, epoch=-1):
    model_path = '../Model'
    files = os.listdir(model_path)
    model_list = []
    for file in files:
        if '.pt' in file and (model_name + '____' + str(test_index)) in file:
            model_list.append(file)

    model_list.sort()
    
    if len(model_list) < 1:
        return None, -1, None, None
    else:
        if epoch != -1:
            model_name = model_name + '____' + str(test_index) + '__' + str(epoch) + '.pt'
            if os.path.isfile(os.path.join(model_path, model_name)):
                model_out = os.path.join(model_path, model_name)
                model_epoch = epoch
            else:
                print 'NO MODEL!!'
                model_out = os.path.join(model_path, model_list[-1])
                model_epoch = int(model_list[-1].split('__')[-1].split('.')[0])
        else:
            model_out = os.path.join(model_path, model_list[-1])
            model_epoch = int(model_list[-1].split('__')[-1].split('.')[0])  



        f = open(model_out.replace('.pt', '.txt'), 'r')
        line = f.readline()
        
        batch_size = int(line.split(',')[0])
        learning_rate = float(line.split(',')[1])
        f.close()

    
        return model_out, model_epoch, batch_size, learning_rate
