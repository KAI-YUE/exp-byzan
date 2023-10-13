from typing import List, Optional

import torch
import numpy as np

from fedlearning.aggregators.initialize import _BaseAggregator
from fedlearning.buffer import _get_para, WeightBuffer

class DnC(_BaseAggregator):
    r"""A robust aggregator from paper `Manipulating the Byzantine: Optimizing
    Model Poisoning Attacks and Defenses for Federated Learning.

    <https://par.nsf.gov/servlets/purl/10286354>`_.
    """

    def __init__(self, config):
        super(DnC, self).__init__(config)

        self.num_byzantine = 10
        self.sub_dim = 10000
        self.num_iters = 5
        self.filter_frac = 1.

    def _get_updates(self, local_packages):
        updates = []
        for i, user_idx in enumerate(local_packages):
            state_dict = local_packages[user_idx].state_dict()
            updates.append(_get_para(state_dict))
            
        updates = torch.stack(updates)
        return updates


    def _aggregate(self, local_packages):
        updates = self._get_updates(local_packages)
        d = len(updates[0])

        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[
                : len(updates) - int(self.filter_frac * self.num_byzantine)
            ]
            benign_ids.append(good)

        # Convert the first list to a set to start the intersection
        intersection_set = set(benign_ids[0])

        # Iterate over the rest of the lists and get the intersection
        for lst in benign_ids[1:]:
            intersection_set.intersection_update(lst)

        # Convert the set back to a list
        benign_ids = list(intersection_set)
        benign_updates = updates[benign_ids, :].mean(dim=0)
        
        shape, num_params = self._retrieve_shape(local_packages[0].state_dict())
        delta = self._updates_to_state_dict(benign_updates, shape, num_params)

        return WeightBuffer(delta)
    
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
