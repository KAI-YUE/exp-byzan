import os
import copy
import torch
import numpy as np

from fedlearning.client import Client
from fedlearning.buffer import WeightBuffer

from deeplearning.utils import test
from deeplearning.datasets import fetch_dataloader


class RandomTrapSetter(Client):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config, model, **kwargs):
        super(RandomTrapSetter, self).__init__(config, model, **kwargs)
        # set up a target model an attacker wants to replace
        self.target_w = WeightBuffer(model.state_dict())
        self.total_users = config.total_users
        self.num_attacker = config.num_attackers
        self.num_benign = self.total_users - self.num_attacker
        self.scaling_factor = config.scaling_factor

        # settings for grid search
        self.steps = 5
        self.distance = config.radius

    def init_local_dataset(self, dataset, data_idx):
        subset = {"images":dataset.dst_train['images'][data_idx], "labels":dataset.dst_train['labels'][data_idx]}
        self.data_loader = fetch_dataloader(self.config, subset, shuffle=True)

    def grid_search(self, network, data_loader, criterion):
        dir_one = WeightBuffer(network.state_dict(), mode="rand")
        dir_two = WeightBuffer(network.state_dict(), mode="rand")
        cursor = WeightBuffer(network.state_dict(), mode="copy")

        # layer-wise normalization 
        for w_name, w_val in dir_one._weight_dict.items():
            dir_one._weight_dict[w_name] = (dir_one._weight_dict[w_name]*cursor._weight_dict[w_name].norm()*self.distance)/(self.steps*dir_one._weight_dict[w_name].norm())
            dir_two._weight_dict[w_name] = (dir_two._weight_dict[w_name]*cursor._weight_dict[w_name].norm()*self.distance)/(self.steps*dir_two._weight_dict[w_name].norm())

        dir_one, dir_two = dir_one*(self.steps/2), dir_two*(self.steps/2)
        cursor = cursor - dir_one
        cursor = cursor - dir_two
        dir_one, dir_two = dir_one*(2/self.steps), dir_two*(2/self.steps)
        start_point = copy.deepcopy(cursor)

        data_matrix = []
        for i in range(self.steps):
            data_column = []

            for j in range(self.steps):
                # column index corresponds to dir_two, row index corresponds to dir_one
                # for every other column, reverse the order in which the column is generated
                # so you can easily use in-place operations to move along dir_two
                if i % 2 == 0:
                    network.load_state_dict(cursor._weight_dict)
                    acc, loss = test(data_loader, network, criterion, self.config)
                    data_column.append(acc)
                    # data_column.append(loss)
                    cursor = cursor + dir_two
                else:
                    network.load_state_dict(cursor._weight_dict)
                    acc, loss = test(data_loader, network, criterion, self.config)
                    data_column.insert(0, acc)
                    # data_column.insert(0, loss)
                    cursor = cursor - dir_two

            data_matrix.append(data_column)
            cursor = cursor + dir_one

        data_matrix = np.asarray(data_matrix)
        low_acc_idx = np.unravel_index(np.argmin(data_matrix), data_matrix.shape)
        
        print(low_acc_idx)
        dir_one, dir_two = dir_one*low_acc_idx[0], dir_two*low_acc_idx[1]
        start_point = start_point + dir_one
        start_point = start_point + dir_two

        network.load_state_dict(start_point._weight_dict)
        acc, loss = test(data_loader, network, criterion, self.config)

        print("Target_low_acc {:.3f}".format(np.min(data_matrix.flatten())))
        print("actual acc {:.3f}".format(acc))

        return start_point

    def set_target_model(self):
        if os.path.exists(self.config.model_checkpoint):
            checkpoint = torch.load(self.config.model_checkpoint)
            self.target_w.push(checkpoint["state_dict"])
        else:
            # randomly set a target model for now
            for w_name, w in self.target_w._weight_dict.items():
                self.target_w._weight_dict[w_name] = torch.rand_like(w)


    def estimate_weight(self, criterion, **kwargs):
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


    def local_step(self, oracle, network, data_loader, criterion, comm_round, **kwargs):
        backup_weight = copy.deepcopy(network.state_dict())

        if comm_round % self.config.change_target_freq == 0:
            # self.estimate_weight(criterion)
            # hypothetical_weight = self.local_model
            # network.load_state_dict(hypothetical_weight.state_dict())
            self.target_w = self.grid_search(network, data_loader, criterion)

        self.interm_w = self.target_w*(self.total_users/self.num_attacker) + oracle*(self.num_benign/(self.num_attacker*self.scaling_factor))
        # self.complete_attack = True

        network.load_state_dict(backup_weight)

    def compute_delta(self):
        delta = (self.w0*(self.total_users/self.num_attacker) - self.interm_w)*self.scaling_factor 
        return delta