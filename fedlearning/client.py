import copy
import numpy as np

from deeplearning.datasets import fetch_dataloader
from deeplearning.utils import init_optimizer, AverageMeter

class LocalUpdater(object):
    def __init__(self, config, model, dataset, data_idx, **kwargs):
        """Construct a local updater for a user.
        """
        
        self.local_model = copy.deepcopy(model)
        self.optimizer = init_optimizer(config, self.local_model)
        self.tau = config.tau
        self.device = config.device

        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(config, subset, shuffle=True)

    def local_step(self, criterion, **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model
        """
        tau_counter = 0
        break_flag = False

        while not break_flag:
            for i, contents in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                target = contents[1].to(self.device)
                input = contents[0].to(self.device)

                # Compute output
                output = self.local_model(input)
                loss = criterion(output, target).mean()

                # Compute gradient and do SGD step
                loss.backward()
                self.optimizer.step()

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break
                
    def uplink_transmit(self):
        """Simulate the transmission of local weights to the central server.
        """ 
        # sample a ternary weight
        local_package = self.local_model.state_dict()

        return local_package