import copy
import numpy as np

# My libraries
from fedlearning.buffer import WeightBuffer
from fedlearning.aggregators import aggregator_registry

from deeplearning.utils import test

class GlobalUpdater(object):
    def __init__(self, config, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:
        """
        self.num_users = int(config.total_users*config.part_rate)
        self.device = config.device

        # self.aggregator = aggregator_registry[config.aggregator](config)
        self.aggregators = [aggregator_registry[agg](config) for agg, _ in aggregator_registry.items()]
        self.aggregator_names = list(aggregator_registry.keys())
        self.data_loader = kwargs["data_loader"]

    def global_step(self, model, benign_packages, attacker_packages, **kwargs):
        # merge benign and attacker packages, as we assume the server does not know which client is attacker
        benign_packages.update(attacker_packages)

        criterion = kwargs["criterion"]
        backup_weight = copy.deepcopy(model.state_dict())
        val_acc = []

        for i, agg in enumerate(self.aggregators):
            accumulated_delta = agg(benign_packages)
            global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
            model.load_state_dict(global_weight.state_dict())
        
            # validate each aggregation result
            acc = test(self.data_loader, model, criterion, self.config)
            val_acc.append(acc)

            model.load_state_dict(backup_weight)
            print("agg {:s} {:.4f}".format(self.aggregator_names[i], acc))

        max_acc_idx = np.argmax(val_acc)
        print("max acc: {:s} {:.4f}".format(self.aggregator_names[max_acc_idx], val_acc[max_acc_idx]))

        accumulated_delta = self.aggregators[max_acc_idx](benign_packages)
        global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
        model.load_state_dict(global_weight.state_dict())

        # model.load_state_dict(accumulated_delta.state_dict())

        # reference = kwargs["reference"]
        # model.load_state_dict(reference.state_dict())

    @property
    def momentum(self):
        # return self.aggregator.momentum
        return None
