import numpy as np
import random



def makeSquare(img_3D, size):
    pad_img = np.zeros((size, size, size))
    pad_img[0:img_3D.shape[0], 0:img_3D.shape[1], 0:img_3D.shape[2]] = img_3D

    return pad_img

def getSmallImage_3D(img3D, size):
    x, y, z = findMidPoint(img3D)
    smallSize = size / 2
    smallImage = img3D[x-smallSize:x+smallSize, x-smallSize:x+smallSize, x-smallSize:x+smallSize]
    
    return smallImage

def rotate_3D(img_3D):
    axis_rand = random.randrange(0, 3)
    rot_rand = random.randrange(0, 4)
    axis_list = [(0, 1), (0, 2), (1, 2)]
    img_rotate = np.rot90(img_3D, rot_rand, axis_list[axis_rand]).copy()

    return img_rotate

def cutAxis(img_3D):
    x, y, z = findMidPoint(img_3D)
    img_X = img_3D[x, :, :]
    img_Y = img_3D[:, y, :]
    img_Z = img_3D[:, :, z]

    return img_X, img_Y, imgZ


def findMidPoint(img_3D):
    x = img_3D.shape[0] / 2
    y = img_3D.shape[1] / 2
    z = img_3D.shape[2] / 2

    return x, y, z

def getSliceIdx(length, ratio):
    start = int((length - (length * ratio)) / 2)
    end = length - start

    return start, end

def randomCrop(img_3D, size):
    rand = random.randrange(0, (img_3D.shape[0] - size))
    croppedImg = img_3D[rand:size + rand, rand:size + rand,rand:size + rand]


    return croppedImg

