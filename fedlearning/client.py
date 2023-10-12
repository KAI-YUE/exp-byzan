import copy
import numpy as np

import torch

from fedlearning.buffer import WeightBuffer
from fedlearning.compressors import compressor_registry
from deeplearning.datasets import fetch_dataloader
from deeplearning.utils import init_optimizer, accuracy
from deeplearning.utils import test

class Client(object):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        self.local_model = copy.deepcopy(model)
        self.w0 =  WeightBuffer(model.state_dict())
        self.optimizer = init_optimizer(config, self.local_model)
        self.tau = config.tau
        self.device = config.device
        self.config = config
        self.complete_attack = False
        self.init_compressor(config)
        self.powerful = False

    def init_local_dataset(self, *args):
        self.data_loader = None

    def init_compressor(self, config):
        if config.compressor in compressor_registry.keys():
            self.compressor = compressor_registry[config.compressor](config)
        else:
            self.compressor = None

    def local_step(self):
        pass

    def uplink_transmit(self):
        delta = self.compute_delta()
        self.postprocessing(delta)
        return delta

    def postprocessing(self, delta):
        """Compress the local gradients.
        """
        if self.compressor is None:
            return
        gradient = delta.state_dict()
        for w_name, grad in gradient.items():
            gradient[w_name] = self.compressor.compress(grad)


class LocalUpdater(Client):
    def __init__(self, config, model, **kwargs):
        """Construct a local updater for a user.
        """
        super(LocalUpdater, self).__init__(config, model, **kwargs)

    def init_local_dataset(self, dataset, data_idx):
        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)

    def local_step(self, criterion, **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model
        """
        # acc, loss = test(self.data_loader, self.local_model, criterion, self.config)
        # print("Client initial acc: {:.3f}".format(acc))

        tau_counter = 0
        break_flag = False

        loss_trajectory, acc_trajectory = [], [] 
        while not break_flag:
            for i, contents in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                target = contents[1].to(self.device)
                input = contents[0].to(self.device)

                # Compute output
                output = self.local_model(input)
                loss = criterion(output, target).mean()
                
                acc = accuracy(output.data, target, topk=(1,))[0]

                # Compute gradient and do SGD step
                loss.backward()

                #local dp setting
                if self.config.ldp:
                    clip(optimizer=self.optimizer, max_norm=self.config.local_bound)
                    add_noise(optimizer=self.optimizer, max_norm=self.config.local_bound, 
                              batch_size=self.config.batch_size, 
                              std=self.config.local_std,
                              device=self.device)

                self.optimizer.step()

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break

                loss_trajectory.append(loss.item())
                acc_trajectory.append(acc.item())

        # loss_trajectory = [np.mean(np.asarray(loss_trajectory))]

        # loss_trajectory.append(loss.item())

        # return the last loss val?
        return loss_trajectory, acc_trajectory

    def compute_delta(self):
        """Simulate the transmission of local gradients to the central server.
        """ 
        w_tau = WeightBuffer(self.local_model.state_dict())
        delta = self.w0 - w_tau

        if self.config.ddp:
            self.ddp(delta)

        return delta
    
    def ddp(self, delta):
        for w_name, w_val in delta._weight_dict.items():
            # clip and add noise
            delta._weight_dict[w_name].data = w_val.clamp(-self.config.bound, self.config.bound) 
            # delta._weight_dict[w_name].data += self.config.std * torch.randn_like(w_val)


def clip(optimizer, max_norm):
    for i, w_val in enumerate(optimizer.param_groups[0]["params"]):
        w_val.grad.data = torch.clamp(w_val.grad.data, -max_norm, max_norm)

def add_noise(optimizer, max_norm, batch_size, std, device):
    for i, w_val in enumerate(optimizer.param_groups[0]["params"]):
        noise = std*torch.randn_like(w_val)
        w_val.grad.data += max_norm * 2 / batch_size * noise