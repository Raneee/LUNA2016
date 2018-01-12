# LUNA2016

# 실행

Train

$ python main.py --TT 'Train' --Model 1 --ImgSize 64 --Epoch 10 --Pretrained True
    
    --TT : 'Train' / 'Test'
    --Model : 0 / 1 / 2 / ... /
    --Batch : 64 / 128 / 256 / ... /
    --ImgSize : 32 / 48 / 64
    --Epoch : None ~ 
    --Pretrained : True / False


Test
    
$ python main.py --TT 'Test' --Model 1 --Epoch 0 --ImgSize 64   --Pretrained True
    
Model

    0 -> Resnet2D
    1 -> 3DNet
    2 -> 2D3DNet
    3 -> Resnet3D
    4 -> Densenet3D
    5 -> Densenet2D
    6 -> ResResnet ??
    
    
nohup - background 실행

$ nohup python main.py --TT 'Train' --Model 1 --ImgSize 64 --Epoch 10 > blah.txt


https://github.com/kenshohara/video-classification-3d-cnn-pytorch

https://github.com/kenshohara/3D-ResNets-PyTorch


