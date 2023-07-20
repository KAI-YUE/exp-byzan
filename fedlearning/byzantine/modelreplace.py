import os
import torch

from fedlearning.client import Client
from fedlearning.buffer import WeightBuffer

class ModelReplaceAttacker(Client):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config, model, **kwargs):
        super(ModelReplaceAttacker, self).__init__(config, model, **kwargs)
        # set up a target model an attacker wants to replace
        self.target_w = WeightBuffer(model.state_dict())
        self.total_users = config.total_users
        self.num_attacker = config.num_attackers
        self.num_benign = self.total_users - self.num_attacker
        self.scaling_factor = config.scaling_factor
        self.set_target_model()

    def set_target_model(self):
        
        if os.path.exists(self.config.model_checkpoint):
            checkpoint = torch.load(self.config.model_checkpoint)
            self.target_w.push(checkpoint["state_dict"])
        else:
            # randomly set a target model for now
            for w_name, w in self.target_w._weight_dict.items():
                self.target_w._weight_dict[w_name] = torch.rand_like(w)

    def local_step(self, oracle, momentum=None, **kwargs):
        self.interm_w = self.target_w*(self.total_users/self.num_attacker) + oracle*(self.num_benign/(self.num_attacker*self.scaling_factor))
        self.complete_attack = True

    def compute_delta(self):
        delta = (self.w0*(self.total_users/self.num_attacker) - self.interm_w)*self.scaling_factor 
        return delta


class DynamicModelReplaceAttacker(Client):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config, model, **kwargs):
        super(ModelReplaceAttacker, self).__init__(config, model, **kwargs)
        # set up a target model an attacker wants to replace
        self.target_w = WeightBuffer(model.state_dict())
        self.total_users = config.total_users
        self.num_attacker = config.num_attackers
        self.num_benign = self.total_users - self.num_attacker
        self.scaling_factor = config.scaling_factor
        

    def set_target_model(self):
        if os.path.exists(self.config.model_checkpoint):
            checkpoint = torch.load(self.config.model_checkpoint)
            self.target_w.push(checkpoint["state_dict"])
        else:
            # randomly set a target model for now
            for w_name, w in self.target_w._weight_dict.items():
                self.target_w._weight_dict[w_name] = torch.rand_like(w)

    def local_step(self, oracle, momentum=None, **kwargs):
        self.interm_w = self.target_w*(self.total_users/self.num_attacker) + oracle*(self.num_benign/(self.num_attacker*self.scaling_factor))
        self.complete_attack = True

    def compute_delta(self):
        delta = (self.w0*(self.total_users/self.num_attacker) - self.interm_w)*self.scaling_factor 
        return delta