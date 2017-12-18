import sys
sys.path.insert(0, 'Tools')
sys.path.insert(0, 'CFG')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import numpy as np
import os
import model as MODEL
import DataLoader as DL
import Tools_Torch as TORCH_T
import Tools_Image as IMG_T
import Tools_IO as IO_T
import Tools_Summary as S_T


def test(model_idx, batch_size):


    f = file('../Output/final.csv', 'a')
    f.write('seriesuid,coordX,coordY,coordZ,probability\n')


    for test_index in range(10):
        print 'Test for ', test_index + 1, ' fold'


        model, model_name, batch_size = TORCH_T.model_setter(model_idx, batch_size)
        model_path, model_epoch, previous_batch_size, previous_learning_rate = IO_T.modelLoader(model_name, test_index)




        model.load_state_dict(torch.load(model_path))
        model.eval()
        print '\nModel Name : ', model_name
        print '\nBatch_size : ', batch_size  



        correct_cnt = 0
        correct_mal = 0
        all_mal = 0

        patientDict = None
        candidateList = None

        patientDict, candidateList = IO_T.makePreLists(train_index, isBalanced=True)

        print '  Patient Count : ', len(patientDict)
        print '  Nodule Count : ', len(noduleCandidate)



        for batch_index in range((len(candidateList) / batch_size) + 1):
            batch_img, batch_label, batch_P_ID, batch_XYZ = DL.makeBatch(batch_index, batch_size, candidateList, patientDict)
            
            img_32 = TORCH_T.to_var(torch.from_numpy(batch_img[0]).float())
            img_48 = TORCH_T.to_var(torch.from_numpy(batch_img[1]).float())
            img_64 = TORCH_T.to_var(torch.from_numpy(batch_img[2]).float())
            img_2D = TORCH_T.to_var(torch.from_numpy(batch_img[3]).float())
            label = TORCH_T.to_var(torch.LongTensor(batch_label).view(-1))



            optimizer.zero_grad()
            if model_idx == 0:
                outputs = model(img_2D)
            elif model_idx == 1:
                outputs = model(img_32, img_48, img_64)
            else:
                outputs = model(img_32, img_48, img_64, img_2D)
            guess, guess_i = IO_T.classFromOutput(outputs)
                    

            correct = np.sum(np.array(guess_i) == (label.data).cpu().numpy())
            correct_cnt += correct
            if batch_index % 100 == 0:
                print '  ', batch_index, ' Batch Accuracy : ', correct * 100 / batch_size, '%'
        print 'Test set (', test_index + 1, ') Accuracy: ', correct_cnt ,'/', len(noduleCandidate), '----->', (correct_cnt * 100 / len(noduleCandidate)) , '%'
        print 
    f.close()

'''
def test(idx, batch_size):
    cand_path = '../Data/CSVFILES/candidates_V2.csv'
    candidate_V2 = IO_T.read_candidates_V2(cand_path)
    f = file('../Output/final.csv', 'a')
    f.write('seriesuid,coordX,coordY,coordZ,probability\n')

    for test_index in range(10):
        print 'Test for ', test_index + 1, ' fold'
        
        
        model, model_name, batch_size = TORCH_T.model_setter(idx, batch_size)
        model_path, model_epoch, previous_batch_size, previous_learning_rate  = IO_T.modelLoader(model_name, test_index) 
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print '\nModel Name : ', model_name
        print '\nBatch_size : ', batch_size        
        
        correct_cnt = 0
        correct_mal = 0
        all_mal = 0
    
    
    
        patientDict = IO_T.makePatientDict(test_index, candidate_V2)
        noduleCandidate = IO_T.makeCandidateList(patientDict, isTest=True)
        print '  Patient Count : ', len(patientDict)
        print '  Nodule Count : ', len(noduleCandidate)
    
        dataset = DL.my_dataset_byInfo()
        dataset.initialize(noduleCandidate, patientDict)
        
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        
        for batch_index, (img_tensor, label_tensor, P_ID, XYZ) in enumerate(test_loader):
            img_32 = TORCH_T.to_var(img_tensor[0], volatile=True)
            img_48 = TORCH_T.to_var(img_tensor[1], volatile=True)
            img_64 = TORCH_T.to_var(img_tensor[2], volatile=True)
            img_2D = TORCH_T.to_var(img_tensor[3], volatile=True)
            label = TORCH_T.to_var(label_tensor.view(-1))
            
            
            if idx == 0:
                outputs = model(img_2D)
            elif idx == 1:
                outputs = model(img_32, img_48, img_64)
            else:
                outputs = model(img_32, img_48, img_64, img_2D)
            lines = IO_T.modify_candidates_V2_OUT(P_ID, XYZ, F.softmax(outputs).data.cpu().numpy())
            for line in lines:
                f.write(str(line))
            guess, guess_i = IO_T.classFromOutput(outputs)


            correct = np.sum(np.array(guess_i) == (label.data).cpu().numpy())
            correct_cnt += correct
            if batch_index % 100 == 0:
                print '  ', batch_index, ' Batch Accuracy : ', correct * 100 / batch_size, '%'
        print 'Test set (', test_index + 1, ') Accuracy: ', correct_cnt ,'/', len(noduleCandidate), '----->', (correct_cnt * 100 / len(noduleCandidate)) , '%'
        print 
    f.close()
'''
