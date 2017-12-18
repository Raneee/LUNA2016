import argparse
import train as Train
import test as Test

parser = argparse.ArgumentParser()
parser.add_argument("--TT", type=str, help="Train???? or Test????")
parser.add_argument("--Model", type=int, help="What Kind of Model????")
parser.add_argument("--GPU", type=int, help="GPU Number????")
parser.add_argument("--Batch", type=int, help="Batch Size????")
parser.add_argument("--Epoch", type=int, help="Epoch Count????")
args = parser.parse_args()


Train_or_Test = args.TT
Model_Type = args.Model
GPU_Number = args.GPU
Batch_Size = args.Batch
Epoch_Cnt = args.Epoch


print 'Train or Test : ', Train_or_Test
print 'Model Type : ', Model_Type
print 'GPU Number : ', GPU_Number
print 'Batch Size : ', Batch_Size
print 'Epoch Count : ', Epoch_Cnt
print 


if Train_or_Test == 'Train':
    for test_idx in range(10):
        Train.train(Model_Type, Epoch_Cnt, test_idx, Batch_Size)
else:
    Test.test(Model_Type, Batch_Size)



