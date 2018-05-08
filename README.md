# Pulmonary Lung Nodule Recognition Using 3D Deep Convolutional Neural Network



## Introduction
Detecting and examining pulmonary nodules early is one of the best ways to prevent lung cancer deaths. For this purpose, accurate nodule recognition is key in diagnosing pulmonary nodules. In this paper, we introduce a shortcut connection model and a dense connection model that use a 3D deep convolutional neural network (DCNN) with a shortcut connection and a dense connection for pulmonary nodule recognition. We also apply ensemble methods to boost performance. The performance of our models is compared with that of the models which were submitted to the false positive reduction track of the [LUng Nodule Analysis 2016 Challenge (LUNA2016)](https://luna16.grand-challenge.org/). Our 3D DCNN model the with ensemble method ESB-All achieved the highest competition performance metric score of 0.910. This result demonstrates that capturing 3D features of nodules is important in improving performance and that our approach of employing deep layers with the shortcut and dense connections is effective for learning such features.

## Requirments

This code has been tested on Ubuntu 16.04 64-bit system.

### Prerequisites

[Python 2](https://www.python.org/download/releases/2.7.2/)

[PyTorch](http://pytorch.org/)

[LUNA Dataset](https://luna16.grand-challenge.org/) 



## Installing Code
1. Clone this repository.
2. Set the dataset path in the codes.




     
     
## Usage
### Args

    --TT : 'Train' / 'Test'                             -> Train or Test
    --Model : 0 / 1 / 2 / ... /                         -> Model Index
    --Batch : default or 64 / 128 / ... /               -> Batch Size
    --ImgSize : 32 / 48 / 64                            -> Input Image Size
    --Epoch : 0 / 1 / ... / ~                           -> Train : How many epochs to run for train 
                                                        -> Test  : Which epoch model for test
### Model Index
    
    0 -> Resnet3D
    1 -> Densenet3D   


### Train

    $ python main.py --TT 'Train' --Model 1  --Epoch 10 --ImgSize 64
    

### Test
    
    $ python main.py --TT 'Test' --Model 1 --Epoch 0 --ImgSize 64


## Contact

Hwejin Jung(hwejin23@gmail.com)


## Acknowledgements

Our code is inspired by the [Video 3D Classification](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) code.

