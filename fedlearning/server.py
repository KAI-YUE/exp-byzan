# My libraries
from fedlearning.buffer import WeightBuffer
from config.utils import *

class GlobalUpdater(object):
    def __init__(self, config, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
        """
        self.num_users = int(config.users*config.part_rate)
        self.global_weight = None    

        self.device = config.device


    def global_step(self, model, local_packages, **kwargs):
        idx = list(local_packages.keys())[0]
        accumulated_weight = WeightBuffer(local_packages[idx], "zeros")
        for user_id, package in local_packages.items():
            accumulated_weight = WeightBuffer(package) + accumulated_weight

        accumulated_weight = accumulated_weight * (1/self.num_users)
        self.global_weight = accumulated_weight.state_dict()
        model.load_state_dict(self.global_weight)
