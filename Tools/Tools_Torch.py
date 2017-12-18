import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
