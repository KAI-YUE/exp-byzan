from typing import Dict
import torch
import random

from fedlearning.client import Client
from fedlearning.buffer import _get_para, WeightBuffer

class MinMax(Client):
    def __init__(self, config, model, **kwargs):
        super(MinMax, self).__init__(config, model, **kwargs)

        self.threshold = 3.0
        self.threshold_diff = 1e-4
        self.num_byzantine = None
        self.negative_indices = None

    def _get_updates(self, local_packages):
        updates = []
        for i, user_idx in enumerate(local_packages):
            state_dict = local_packages[user_idx].state_dict()
            updates.append(_get_para(state_dict))
            
        updates = torch.stack(updates)
        return updates

    def local_step(self, benign_packages, network, data_loader, criterion, comm_round, **kwargs):
        benign_updates = self._get_updates(benign_packages)
        mean_grads = benign_updates.mean(dim=0)
        deviation = benign_updates.std(dim=0)
        threshold = torch.cdist(benign_updates, benign_updates, p=2).max()

        low = 0
        high = 5
        while abs(high - low) > 0.01:
            mid = (low + high) / 2
            mal_update = torch.stack([mean_grads - mid * deviation])
            loss = torch.cdist(mal_update, benign_updates, p=2).max()
            if loss < threshold:
                low = mid
            else:
                high = mid
        
        delta = mean_grads - mid * deviation
        shape, num_params = self._retrieve_shape(benign_packages[0].state_dict())
        self.delta = self._updates_to_state_dict(delta, shape, num_params)

        self.complete_attack = True

    def compute_delta(self):
        return WeightBuffer(self.delta)
    
    def _retrieve_shape(self, state_dict):
        shape = {}
        num_params = []
        for i, key in enumerate(state_dict):
            shape[key] = state_dict[key].shape
            num_params.append(state_dict[key].numel())

        return shape, num_params
    
    def _updates_to_state_dict(self, updates, shape, num_params):
        state_dict = {}
        num_params.append(-1)
        p, q = 0, num_params[0]
        for i, key in enumerate(shape):
            state_dict[key] = updates[p:q].reshape(shape[key])
            p = q
            q += num_params[i+1]

        return state_dict
