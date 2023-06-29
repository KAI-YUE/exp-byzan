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
    data_loader = DataLoader(MyDataset(data["images"], data["labels"]), 
                        shuffle=shuffle, batch_size=config.batch_size)

    return data_loader


# def fetch_dataloader(config, data, shuffle=True):
#     train_loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=shuffle,
#                                                 num_workers=4 ,pin_memory=False)

#     return train_loader