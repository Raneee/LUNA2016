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
    out_file_dir = os.path.join('../Output', 'nopretrain')



    if not os.path.isdir(out_file_dir):
        os.mkdir(out_file_dir)
    out_file_path = os.path.join(out_file_dir, out_name + '.csv')
    print 'Out File : ' , out_file_path, 
    if os.path.isfile(out_file_path):
        os.remove(out_file_path)



    f = file(out_file_path, 'a')
    f.write('seriesuid,coordX,coordY,coordZ,probability\n')

    for test_index in range(10):
        print 'Test for ', test_index + 1, ' fold'

        model, model_name, batch_size = MODEL_T.model_setter(model_idx, img_size=img_size, batch_size=batch_size, isTest=True)
        model_path, model_epoch = MODEL_T.modelLoader(model_name, test_index, img_size, epoch=num_epoch)



        model.load_state_dict(torch.load(model_path))
        model.eval()
        print '\nModel Name : ', model_name
        print '\nModel Path : ', model_path
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

            label = TORCH_T.to_var(torch.LongTensor(batch_label).view(-1))
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
    results_filename = out_file_path
    outputDir = os.path.join(out_file_dir, out_name)
    os.makedirs(outputDir)


    noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir)
    print 'DONE'
