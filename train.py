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
import Tools_IO as IO_T
import Tools_Summary as SUMMARY_T
import Tools_Model as MODEL_T



def train(model_idx, test_index, batch_size, img_size, isContinue=False):

    
    model, model_name, batch_size = MODEL_T.model_setter(model_idx, img_size, batch_size)
    model_path, model_epoch, previous_batch_size, previous_learning_rate = MODEL_T.modelLoader(model_name, test_index, img_size)

 

    print '\nModel Name : ', model_name
    print '\nBatch_Size : ', batch_size
    print '\nTest_Index : ', test_index
    print 
    if isContinue and model_path != None:
        model.load_state_dict(torch.load(model_path))
        print 'Previous Model Loaded!     -> ', model_path
        print 'Start Epoch : ' , model_epoch 
        learning_rate = previous_learning_rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  
        epoch = model_epoch + 1
    else:
        print 'No Model Loaded!'
        print 'Start Epoch : 0'	
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  
        epoch = 0        




    for train_index in range(10):
        patientDict = None
        candidateList = None
        if train_index != test_index:
            train_correct_cnt = 0
            print '      Train for ', train_index + 1, ' fold'
            patientDict, candidateList = IO_T.makePreLists(train_index, isBalanced=True)

            print '          Patient Count : ', len(patientDict)
            print '          Nodule Count : ', len(candidateList)
    
            for batch_index in range((len(candidateList) / batch_size)):
                batch_img, batch_label, batch_P_ID, batch_XYZ = DL.makeBatch(batch_index, batch_size, candidateList, patientDict)




                label = TORCH_T.to_var(torch.LongTensor(batch_label).view(-1))
                
                optimizer.zero_grad()
                outputs = model(TORCH_T.imageOnTorch(batch_img, model_idx, img_size=img_size))


                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()
                guess, guess_i = IO_T.classFromOutput(outputs)
                    

                if batch_index % 100 == 0:
                    print '        In mini-batch ', batch_index
                    print '                   Loss : ', loss.data[0]
                    TP, FP, FN, TN = SUMMARY_T.result_Summary(guess_i, label, isPrint=True)
                    correct = SUMMARY_T.result_correct(guess_i, label, isPrint=True)
                else:
                    TP, FP, FN, TN = SUMMARY_T.result_Summary(guess_i, label)
                    correct = SUMMARY_T.result_correct(guess_i, label)
                train_correct_cnt += correct
                    



            print train_correct_cnt, '/', len(candidateList), '----->', (train_correct_cnt * 100 / len(candidateList)) , '%'
    print 'Model Stored ----------->   ' , model_name + '____' + str(test_index)+ '__'+ str(epoch) + '__' + str(img_size) + '.pt'
    torch.save(model.state_dict(), '../Model/' + model_name + '____' + str(test_index)+ '__'+ str(epoch) + '__' + str(img_size) + '.pt')
    save_rate = 0.001
    for param_group in optimizer.param_groups:
        save_rate = param_group['lr']            
    f = open('../Model/' + model_name + '____' + str(test_index)+ '__'+ str(epoch) + '__' + str(img_size) + '.txt', 'w')
    f.write(str(batch_size) +',' + str(save_rate))
    f.close()  

