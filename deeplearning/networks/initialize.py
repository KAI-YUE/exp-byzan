import os
# PyTorch Libraries
import torch.nn as nn

# My Libraries
from deeplearning.networks.lenet import LeNet, LeNet_5

def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

nn_registry = {
    # "resnet18":         ResNet18,

    "lenet":            LeNet,
    "lenet_5":          LeNet_5,
}
