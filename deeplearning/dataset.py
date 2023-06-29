import os
import numpy as np
import pickle
from PIL import Image

# PyTorch Libraries
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from config.utils import *

def assign_user_resource(config, userID, user_with_data, resource, **kwargs):
    """Simulate one user resource by assigning the dataset and configurations.
    """
    user_resource = {}

    user_resource["lr"] = config.lr
    user_resource["momentum"] = config.momentum
    user_resource["weight_decay"] = config.weight_decay
    user_resource["device"] = config.device
    user_resource["batch_size"] = config.batch_size

    if resource is not None:
        user_resource["images"] = resource["train_data"]["images"]
        user_resource["labels"] = resource["train_data"]["labels"]

    # shuffle the sampleIDs
    np.random.shuffle(user_with_data[userID])

    user_resource["user_with_data"] = user_with_data[userID]

    return user_resource

