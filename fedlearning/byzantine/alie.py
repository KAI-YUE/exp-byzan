import numpy as np
from scipy.stats import norm

import torch

from fedlearning.client import Client
from fedlearning.buffer import WeightBuffer

class AlieAttacker(Client):
    def __init__(self, config, model, **kwargs):
        super(AlieAttacker, self).__init__(config, model, **kwargs)
        s = np.floor(config.total_users / 2) - config.num_attackers
        cdf_value = (config.total_users - config.num_attackers - s) / (config.total_users - config.num_attackers)
        self.z_max = norm.ppf(cdf_value)

    def local_step(self, benign_packages, **kwargs):
        layerwise_params = {}
        package_keys = list(benign_packages.keys())
        user_state_dict = benign_packages[package_keys[0]].state_dict()

        for param_name, param in user_state_dict.items():
            layerwise_params[param_name] = []

        for param_name, param in user_state_dict.items():
            for user_id, package in benign_packages.items():
                param = package.state_dict()[param_name].unsqueeze(0)
                layerwise_params[param_name].append(param)

        mu, std = {}, {}
        for param_name, param in user_state_dict.items():
            param_matrix = torch.cat(layerwise_params[param_name], dim=0)
            mu[param_name] = torch.mean(param_matrix, dim=0)
            std[param_name] = torch.std(param_matrix, dim=0)

        malicious_package = WeightBuffer(user_state_dict, mode="zero")
        for param_name, param in user_state_dict.items():
            malicious_package._weight_dict[param_name] = mu[param_name] - self.z_max*std[param_name]

        self.delta = malicious_package
        self.complete_attack = True
        
    def compute_delta(self):
        return self.delta