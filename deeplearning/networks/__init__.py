# My Libraries
from deeplearning.networks.lenet import LeNet, LeNet_5
from deeplearning.networks.harnet import HARnet

nn_registry = {
    "lenet":            LeNet,
    "lenet_5":          LeNet_5,

    "harnet":           HARnet,
}
