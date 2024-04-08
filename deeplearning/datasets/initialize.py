import copy
import numpy as np

from deeplearning.datasets.cifar10 import CIFAR10
from deeplearning.datasets.fmnist import FashionMNIST
from deeplearning.datasets.ucihar import UCI_HAR

dataset_registry = {
    "cifar10": CIFAR10,
    "fmnist": FashionMNIST,
    
    "uci_har": UCI_HAR,
}

def fetch_dataset(config):
    dataset = dataset_registry[config.dataset](config.data_path)

    config.num_classes = dataset.num_classes
    config.im_size = dataset.im_size
    config.channel = dataset.channel

    return dataset


def fetch_subset(dataset, size=1000, config=None):
    # if we use har dataset, we just pick one user 
    # if "har" in config.dataset: 
    #     indices = np.arange(len(dataset[0]))
    #     np.random.shuffle(indices)
    #     rand_indices = indices[:size]

    #     # 0 if the image, 1 if the label; the second 0 is the user id
    #     images, labels = dataset[0][0][rand_indices], dataset[1][0][rand_indices]
        
    #     subset = {}
    #     subset["images"], subset["labels"] = images, labels

    # else:
    indices = np.arange(len(dataset["images"]))
    np.random.shuffle(indices)
    rand_indices = indices[:size]

    images, labels = dataset["images"][rand_indices], dataset["labels"][rand_indices]

    subset = {}
    subset["images"], subset["labels"] = images, labels

    return subset