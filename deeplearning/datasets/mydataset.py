import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, images, labels):
        """Construct a customized dataset
        """
        if min(labels) < 0:
            labels = (labels).reshape((-1,1)).astype(np.float32)
        else:
            labels = (labels).astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return (image, label)

def fetch_dataloader(config, data, shuffle=True):
    # if "har" in config.dataset:
    #     # if it is a list, we do not fetch the training set
    #     if type(data) == list:
    #         pass
        
    #     return None

    if "cifar" in config.dataset:
        data_loader = DataLoader(data, 
                    batch_size=config.batch_size, shuffle=True,
                    num_workers=config.workers, pin_memory=False)
    
    else:
        data_loader = DataLoader(MyDataset(data["images"], data["labels"]), 
                    shuffle=shuffle, batch_size=config.batch_size)

    return data_loader


def fetch_attacker_dataloader(config, data, attacker_ids, user_data_mapping, shuffle=True):
    if len(attacker_ids) == 0:
        return None
    
    data_idx = []
    for i, id in enumerate(attacker_ids):
        data_idx.extend(user_data_mapping[id]) 


    if "cifar" in config.dataset:
        subset = torch.utils.data.Subset(data, data_idx)
        data_loader = DataLoader(subset, 
                    batch_size=config.batch_size, shuffle=True,
                    num_workers=config.workers, pin_memory=False)
    
    else:
        data_loader = DataLoader(MyDataset(data["images"][data_idx], data["labels"][data_idx]),
                            shuffle=shuffle, batch_size=config.batch_size)

    return data_loader