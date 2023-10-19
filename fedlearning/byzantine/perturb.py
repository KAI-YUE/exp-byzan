import os
import copy
import torch
import numpy as np

from fedlearning.client import Client
from fedlearning.buffer import WeightBuffer

from deeplearning.utils import test
from deeplearning.datasets import fetch_dataloader    

class Perturb(Client):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config, model, **kwargs):
        super(Perturb, self).__init__(config, model, **kwargs)
        # set up a target model an attacker wants to replace
        self.target_w = WeightBuffer(model.state_dict())
        self.total_users = config.total_users
        self.num_attacker = config.num_attackers
        self.num_benign = self.total_users - self.num_attacker
        self.scaling_factor = config.scaling_factor

    def init_local_dataset(self, dataset, data_idx):
        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)

    def local_step(self, oracle, network, data_loader, criterion, comm_round, **kwargs):
        backup_weight = copy.deepcopy(network.state_dict())

        acc, loss = test(data_loader, network, criterion, self.config)
        print("Initial acc: {:.3f}".format(acc))

        # approximate the oracle
        # network.load_state_dict(self.target_w.state_dict())
        # oracle = self.estimate_oracle(criterion, data_loader)

        if comm_round % self.config.change_target_freq == 0:
            # noise = WeightBuffer(network.state_dict(), mode="rand")
            # noise = noise * (1.e-2)
            # self.target_w = noise + WeightBuffer(network.state_dict(), mode="copy")
            self.target_w = WeightBuffer(network.state_dict(), mode="copy")

        # network.load_state_dict(self.target_w._weight_dict)
        # acc, loss = test(data_loader, network, criterion, self.config)
        # print("Initial acc: {:.3f}".format(acc))

        network.load_state_dict(backup_weight)

        self.interm_w = self.target_w*(self.total_users/self.num_attacker) + oracle*(self.num_benign/(self.num_attacker*self.scaling_factor))
        self.complete_attack = True

        network.load_state_dict(backup_weight)

    def estimate_oracle(self, criterion, data_loader=None):
        if data_loader is None:
            data_loader = self.data_loader

        tau_counter = 0
        break_flag = False

        # loss_trajectory, acc_trajectory = [], [] 
        while not break_flag:
            for i, contents in enumerate(data_loader):
                self.optimizer.zero_grad()
                target = contents[1].to(self.device)
                input = contents[0].to(self.device)

                # Compute output
                output = self.local_model(input)
                loss = criterion(output, target).mean()
                
                # acc = accuracy(output.data, target, topk=(1,))[0]

                # Compute gradient and do SGD step
                loss.backward()

                self.optimizer.step()

                tau_counter += 1
                if tau_counter >= self.tau:
                    break_flag = True
                    break

                # loss_trajectory.append(loss.item())
                # acc_trajectory.append(acc.item())

        # # return the last loss val?
        # return loss_trajectory, acc_trajectory
        w_tau = WeightBuffer(self.local_model.state_dict())
        delta = self.w0 - w_tau

        return delta

    def compute_delta(self):
        delta = (self.w0*(self.total_users/self.num_attacker) - self.interm_w)*self.scaling_factor 
        return delta