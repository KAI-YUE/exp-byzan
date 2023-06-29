import copy

import torch
import torch.nn as nn

from deeplearning.datasets import fetch_dataloader 

class LocalUpdater(object):
    def __init__(self, config, dataset, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarry):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - mode (int):       the mode indicating the local model type.
                - device (str):     set 'cuda' or 'cpu' for the user. 
        """
        
        self.lr = config.lr
        self.momentum = config.lr
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        self.device = config.device

        self.local_weight = None        
        self.sample_loader = fetch_dataloader(config, dataset)

        self.criterion = nn.CrossEntropyLoss()

        self.tau = config.tau

    def local_step(self, model, **kwargs):
        """Perform local update tau times.
        Args,
            model(nn.module):       the global model.
        """
        # if we are training a full precision network
        # the copy of the model is state dict
        w_copy = copy.deepcopy(model.state_dict())
        # optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        tau_counter = 0
        break_flag = False

        while not break_flag:
            for sample in self.sample_loader:
                image = sample[0].to(self.device)
                label = sample[1].to(self.device)
                optimizer.zero_grad()

                output = model(image)
                loss = self.criterion(output, label)

                loss.backward()
                optimizer.step()

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break
               
        self.local_weight = copy.deepcopy(model.state_dict())
        model.load_state_dict(w_copy)

    def uplink_transmit(self):
        """Simulate the transmission of local weights to the central server.
        """ 
        # sample a ternary weight
        local_package = self.local_weight

        return local_package