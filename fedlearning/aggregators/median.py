import torch

from fedlearning.aggregators.initialize import _BaseAggregator
from fedlearning.buffer import _get_para, WeightBuffer

class Median(_BaseAggregator):
    def __init__(self, config):
        super(Median, self).__init__(config)
        self.num_users = config.total_users

    def _aggregate(self, local_packages):
        updates = self._get_updates(local_packages)
        median_updates, _ = torch.median(updates, dim=0)

        shape, num_params = self._retrieve_shape(local_packages[0].state_dict())
        median_delta = self._updates_to_state_dict(median_updates, shape, num_params)

        return WeightBuffer(median_delta)
    
    def _get_updates(self, local_packages):
        updates = []
        for i, user_idx in enumerate(local_packages):
            state_dict = local_packages[user_idx].state_dict()
            updates.append(_get_para(state_dict))
            
        updates = torch.stack(updates)
        return updates
    
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

