import numpy as np
from scipy.stats import norm

import torch

from fedlearning.client import Client

class AlieAttacker(Client):
    def __init__(self, config, model, **kwargs):
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
            mu = torch.mean(param_matrix, dim=0)
            std = torch.std(param_matrix, dim=0)

        malicious_packages = {}
        for user_id, package in benign_packages.items():
            malicious_packages[user_id] = package.state_dict()
            for param_name, param in package.state_dict().items():
                malicious_packages[param_name] = mu - self.z_max*std

        self.complete_attack = True

        return malicious_packages