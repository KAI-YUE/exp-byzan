# My Libraries
from deeplearning.networks.lenet import LeNet, LeNet_5
from deeplearning.networks.harnet import HARnet
from deeplearning.networks.vit import ViT

nn_registry = {
    "lenet":            LeNet,
    "lenet_5":          LeNet_5,

    "harnet":           HARnet,
    "vit":              ViT,
}
