# Pulmonary Lung Nodule Recognition Using 3D Deep Convolutional Neural Network

This repository contains a code for Pulmonary Lung Nodule Recognition Using 3D Deep Convolutional Neural Network paper.

3D CNNs with shortcut connection and with dense connection is implemented.



## Requirments

[PyTorch](http://pytorch.org/)

[Python 2](https://www.python.org/download/releases/2.7.2/)

[LUNA Dataset](https://luna16.grand-challenge.org/) 


## Model Index

    0 -> Resnet2D           -> remove!!
    1 -> 3DNet              -> remove!!
    2 -> 2D3DNet            -> remove!!
    3 -> Resnet3D
    4 -> Densenet3D
    5 -> Densenet2D         -> remove!!
    6 -> ResResnet??        -> remove!!
    
    
## Directory
    ---Code
     |-Model
     |-Output
     |-Data--------ALLINONE
                 |-CSVFILES
                 |-SUBSET

     
     
## Usage
### Args
    --TT : 'Train' / 'Test'                             -> Train or Test
    --Model : 0 / 1 / 2 / ... /                         -> Model Index
    --Batch : default or 64 / 128 / ... /               -> Batch Size
    --ImgSize : 32 / 48 / 64                            -> Input Image Size
    --Epoch : 0 / 1 / ... / ~                           -> Train : How many epochs to run for train 
                                                        -> Test  : Which epoch model for test



### Train

    $ python main.py --TT 'Train' --Model 1  --Epoch 10 --ImgSize 64
    


### Test
    
    $ python main.py --TT 'Test' --Model 1 --Epoch 0 --ImgSize 64


## Reference Code

[PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial)

[Video 3D Classification](https://github.com/kenshohara/video-classification-3d-cnn-pytorch)


