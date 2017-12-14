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




def train(idx, batch_size):
    cand_path = '../Data/CSVFILES/candidates_V2.csv'
    candidate_V2 = IO_T.read_candidates_V2(cand_path)

    num_epoch = 10
    for epoch in range(num_epoch):
        for test_index in range(10):

          

            model, model_name, batch_size = TORCH_T.model_setter(idx, batch_size)
            model_path, model_epoch = IO_T.modelLoader(model_name, test_index)


            learning_rate = 0.001 / (2 * model_epoch)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  
            
            print '\nModel Name : ', model_name
            print '\nBatch_size : ', batch_size

            if model_epoch != -1:
                model.load_state_dict(torch.load(model_path))
                print 'Previous Model Loaded!     -> ', model_path
                print 'Start Epoch : ' , model_epoch	
            else:
                print 'No Model Loaded!'
                print 'Start Epoch : 0'	
            print epoch, ' / ', num_epoch, ' epoch'
            
            for train_index in range(10):
                if train_index != test_index:

                    train_correct_cnt = 0
                    print '      Train for ', train_index + 1, ' fold'
                    
                    
                    patientDict = IO_T.makePatientDict(train_index, candidate_V2)
                    balancedCandidate = IO_T.makeBalancedList(patientDict)
                    
                    print '          Patient Count : ', len(patientDict)
                    print '          Nodule Count : ', len(balancedCandidate)
                
                    
                    

                    dataset = DL.my_dataset_byInfo()
                    dataset.initialize(balancedCandidate, patientDict)
                    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
                    for batch_index, (img_tensor, label_tensor, P_ID, XYZ) in enumerate(train_loader):
                        img_32 = TORCH_T.to_var(img_tensor[0])
                        img_48 = TORCH_T.to_var(img_tensor[1])
                        img_64 = TORCH_T.to_var(img_tensor[2])
                        img_2D = TORCH_T.to_var(img_tensor[3])
                        label = TORCH_T.to_var(label_tensor.view(-1))

                        optimizer.zero_grad()
                        if idx == 0:
                            outputs = model(img_2D)
                        elif idx == 1:
                            outputs = model(img_32, img_48, img_64)
                        else:
                            outputs = model(img_32, img_48, img_64, img_2D)
                        loss = criterion(outputs, label)

                        loss.backward()
                        optimizer.step()
                        guess, guess_i = IO_T.classFromOutput(outputs)

                        correct = np.sum(np.array(guess_i) == (label.data).cpu().numpy())
                        
                        train_correct_cnt += correct
                        


                        if batch_index % 100 == 0:
                            #torch.save(model.state_dict(), '../Model/' + model_name + '__' + str(test_index)+ '__'+ str(model_epoch + 1) + '.pt')
                            print '        In mini-batch ', batch_index
                            print '                   Accuracy : ', correct ,'/', label_tensor.size()[0], '----->', (correct * 100 / label_tensor.size()[0]) , '%'
                            print '                   Loss : ', loss.data[0]
                            TP, FP, FN, TN = IO_T.result_Summary(np.array(guess_i), (label.data).cpu().numpy())
                            print '                   TP : ', TP, ' FP : ', FP, ' FN : ', FN, ' TN : ', TN

                    print train_correct_cnt, '/', len(balancedCandidate), '----->', (train_correct_cnt * 100 / len(balancedCandidate)) , '%'
                
            torch.save(model.state_dict(), '../Model/' + model_name + '____' + str(test_index)+ '__'+ str(model_epoch + 1) + '.pt')    
