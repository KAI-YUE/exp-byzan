import copy
import numpy as np

from fedlearning.buffer import WeightBuffer
from fedlearning.client import Client
from deeplearning.datasets import fetch_dataloader
from deeplearning.utils import init_optimizer

class OmniscientSignflippingAttacker(Client):
    def __init__(self, config, model, **kwargs):
        super(OmniscientSignflippingAttacker, self).__init__(config, model, **kwargs)
    
    def init_local_dataset(self, *args):
        pass

    def local_step(self, oracle, **kwargs):
        self.delta = oracle * (-1.5)

    def uplink_transmit(self):
        return self.delta


class SignflippingAttacker(Client):
    def __init__(self, config, model, **kwargs):
        super(SignflippingAttacker, self).__init__(config, model, **kwargs)

    def init_local_dataset(self, dataset, data_idx):
        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)

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
                
    def compute_delta(self):
        """Simulate the transmission of local gradients with flipped signs.
        """ 
        w_tau = WeightBuffer(self.local_model.state_dict())
        delta = w_tau - self.w0

        return delta