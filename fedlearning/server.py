# My libraries
from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators import aggregator_registry

class GlobalUpdater(object):
    def __init__(self, config, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
        """
        self.num_users = int(config.total_users*config.part_rate)
        self.device = config.device

        self.aggregator = aggregator_registry[config.aggregator](config)

    def global_step(self, model, benign_packages, attacker_packages, **kwargs):
        # merge benign and attacker packages, as we assume the server does not know which client is attacker
        benign_packages.update(attacker_packages)

        accumulated_delta = self.aggregator(benign_packages)
        global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
        model.load_state_dict(global_weight.state_dict())
        # model.load_state_dict(accumulated_delta.state_dict())

        reference = kwargs["reference"]
        model.load_state_dict(reference.state_dict())

    @property
    def momentum(self):
        return self.aggregator.momentum
