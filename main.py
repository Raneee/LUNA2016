import argparse
import train as Train
import test as Test

parser = argparse.ArgumentParser()
parser.add_argument("--TT", type=str, help="Train???? or Test????")
parser.add_argument("--Model", type=int, help="What Kind of Model????")
parser.add_argument("--Batch", type=int, help="Batch Size????")
parser.add_argument("--Epoch", type=int, help="Epoch Count????")
parser.add_argument("--ImgSize", type=int, help="Image Size????")
#parser.add_argument("--Pretrained", default=False, help="Pretrained Weight???")
#parser.add_argument("--Undersampling", default=False, help="Under Sampling???")
args = parser.parse_args()


'''
def str2bool(v):
    bool_list = ['True']
    if v in bool_list:
        return True
    else:
        return False
'''



Train_or_Test = args.TT
Model_Type = args.Model
Batch_Size = args.Batch
Epoch_Cnt = args.Epoch
Img_Size = args.ImgSize
#Pretrained_Weight = str2bool(args.Pretrained)
#Under_Sampling = str2bool(args.Undersampling)

print 'Train or Test : ', Train_or_Test
print 'Model Type : ', Model_Type
print 'Batch Size : ', Batch_Size
print 'Epoch Count : ', Epoch_Cnt
print 'Image Size : ', Img_Size
#print 'Pretrained Weight : ', Pretrained_Weight
#print 'Under Sampling : ', Under_Sampling
print 


if Train_or_Test == 'Train':
    for test_idx in range(10):
        #Train.train(Model_Type, test_idx, Batch_Size, Img_Size, Pretrained_Weight, under_sampling=Under_Sampling, isContinue=True)
        #Train.train(Model_Type, test_idx, Batch_Size, Img_Size, Pretrained_Weight, isContinue=True)
        Train.train(Model_Type, test_idx, Batch_Size, Img_Size, isContinue=True)
else:
    #Test.test(Model_Type, Epoch_Cnt, Batch_Size, Img_Size, Pretrained_Weight)
    Test.test(Model_Type, Epoch_Cnt, Batch_Size, Img_Size)
