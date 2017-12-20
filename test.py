import sys
sys.path.insert(0, 'Tools')
sys.path.insert(0, 'CFG')
sys.path.insert(0, 'Scripts')


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
import Tools_Summary as SUMMARY_T
import Tools_Model as MODEL_T
from noduleCADEvaluationLUNA16 import *

def test(model_idx, num_epoch, batch_size, img_size):
    out_name = MODEL_T.model_names[model_idx] + '_' + str(num_epoch) + '_' + str(img_size)
    out_file_path = '../Output/' + out_name + '.csv'
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    f = file(out_file_path, 'a')
    f.write('seriesuid,coordX,coordY,coordZ,probability\n')


    for test_index in range(10):
        print 'Test for ', test_index + 1, ' fold'


        model, model_name, batch_size = MODEL_T.model_setter(model_idx, batch_size)
        model_path, model_epoch, previous_batch_size, previous_learning_rate = MODEL_T.modelLoader(model_name, test_index, num_epoch)




        model.load_state_dict(torch.load(model_path))
        model.eval()
        print '\nModel Name : ', model_name
        print '\nBatch_size : ', batch_size  



        correct_cnt = 0
        correct_mal = 0
        all_mal = 0

        patientDict = None
        candidateList = None

        patientDict, candidateList = IO_T.makePreLists(test_index, isTest=True)

        print '  Patient Count : ', len(patientDict)
        print '  Nodule Count : ', len(candidateList)



        for batch_index in range((len(candidateList) / batch_size) + 1):
            batch_img, batch_label, batch_P_ID, batch_XYZ = DL.makeBatch(batch_index, batch_size, candidateList, patientDict)
            '''
            img_32 = TORCH_T.to_var(torch.from_numpy(batch_img[0]).float())
            img_48 = TORCH_T.to_var(torch.from_numpy(batch_img[1]).float())
            img_64 = TORCH_T.to_var(torch.from_numpy(batch_img[2]).float())
            img_2D = TORCH_T.to_var(torch.from_numpy(batch_img[3]).float())
            label = TORCH_T.to_var(torch.LongTensor(batch_label).view(-1))



            if model_idx == 0:
                outputs = model(img_2D)
            elif model_idx == 1:
                outputs = model(img_32, img_48, img_64)
            elif model_idx == 2:
                outputs = model(img_32, img_48, img_64, img_2D)
            else:
                if img_size == 32:
                    convert_img = img_32
                elif img_size == 64:
                    convert_img = img_64
                else:
                    convert_img = img_48
                if convert_img.size()[1] == 1:
                    convert_img = convert_img.data.cpu().numpy()
                    convert_img = np.concatenate((convert_img, convert_img, convert_img), axis = 1) 
                    convert_img = TORCH_T.to_var(torch.from_numpy(convert_img).float())
                outputs = model(convert_img)
            '''
            outputs = model(TORCH_T.imageOnTorch(batch_img, model_idx, img_size=img_size))
            guess, guess_i = IO_T.classFromOutput(outputs)
            lines = IO_T.modify_candidates_V2_OUT(batch_P_ID, batch_XYZ, F.softmax(outputs).data.cpu().numpy())
            for line in lines:
                f.write(str(line))                    

            correct = np.sum(np.array(guess_i) == (label.data).cpu().numpy())
            correct_cnt += correct
            if batch_index % 100 == 0:
                print '  ', batch_index, ' Batch Accuracy : ', correct * 100 / batch_size, '%'
        print 'Test set (', test_index + 1, ') Accuracy: ', correct_cnt ,'/', len(candidateList), '----->', (correct_cnt * 100 / len(candidateList)) , '%'
        print 
    f.close()




    annotations_filename = 'Scripts/annotations/annotations.csv'
    annotations_excluded_filename = 'Scripts/annotations/annotations_excluded.csv'
    seriesuids_filename = 'Scripts/annotations/seriesuids.csv'
    results_filename = '../Output/' + out_name + '.csv'
    outputDir = os.path.join('../Output/', out_name)
    os.makedirs(outputDir)


    noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir)
    print 'DONE'
