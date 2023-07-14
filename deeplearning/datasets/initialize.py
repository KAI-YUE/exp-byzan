
from deeplearning.datasets.cifar10 import CIFAR10
from deeplearning.datasets.fmnist import FashionMNIST

dataset_registry = {
    "cifar10": CIFAR10,
    "fmnist": FashionMNIST,
}


def fetch_dataset(config):
    dataset = dataset_registry[config.dataset](config.data_path)

    config.num_classes = dataset.num_classes
    config.im_size = dataset.im_size
    config.channel = dataset.channel

    return dataset

