import torch

from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators.initialize import _BaseAggregator

class TrimmedMean(_BaseAggregator):
    def __init__(self, config):
        super(TrimmedMean, self).__init__(config)
        self.num_users = config.total_users
        self.b = config.num_attackers

    def _aggregate(self, local_packages):
        neg_smallest, largest, updates, new_stacked = {}, {}, {}, {}
        package_keys = list(local_packages.keys())
        user_state_dict = local_packages[package_keys[0]].state_dict()

        for param_name, param in user_state_dict.items():
            updates[param_name] = []

        for param_name, _ in user_state_dict.items():
            for user_id, package in local_packages.items():
                param = package.state_dict()[param_name].unsqueeze(0)
                updates[param_name].append(param)

        # layer wise topk
        for param_name, param in user_state_dict.items():
            layerwise_update = torch.cat(updates[param_name], dim=0)
            largest[param_name], _ = torch.topk(layerwise_update, k=self.b, dim=0)
            neg_smallest[param_name], _ = torch.topk(-layerwise_update, k=self.b, dim=0)

        for param_name, param in user_state_dict.items():
            layerwise_update = torch.cat(updates[param_name], dim=0)
            new_stacked[param_name] = torch.cat([-largest[param_name], neg_smallest[param_name], layerwise_update], dim=0).sum(dim=0)

        accumulated_delta = WeightBuffer(new_stacked)
        accumulated_delta = accumulated_delta * (1/(self.num_users - 2*self.b))

        return accumulated_delta
