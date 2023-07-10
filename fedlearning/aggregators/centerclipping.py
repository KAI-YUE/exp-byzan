import torch

from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators.initialize import _BaseAggregator

class CenteredClipping(_BaseAggregator):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config):
        self.num_users = config.total_users
        self.cc_bound = config.cc_bound
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.cc_bound / v_norm)
        return v * scale

    def __call__(self, local_packages):
        # if momentum is None, initialize 
        if self.momentum is None:
            self.momentum = WeightBuffer(local_packages[0].state_dict())

        clipped = WeightBuffer(local_packages[0].state_dict(), mode="zero")
        clipped_state_dict = clipped.state_dict()
        momentum_state_dict = self.momentum.state_dict()

        package_keys = list(local_packages.keys())
        user_state_dict = local_packages[package_keys[0]].state_dict()

        for param_name, _ in user_state_dict.items():
            for user_id, package in local_packages.items():
                clipped_update = self.clip(package._weight_dict[param_name] - momentum_state_dict[param_name])
                clipped_state_dict[param_name] += clipped_update

        clipped = clipped * (1/self.num_users)
        self.momentum = self.momentum + clipped

        return self.momentum