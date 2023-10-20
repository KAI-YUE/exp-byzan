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
        
        if config.aggregator == "hybrid":
            self.hybrid = True
            # agg_candidates = ["mean", "median", "krum", "trimmed_mean" ,"centeredclipping", "signguard"]
            agg_candidates = ["median", "krum", "trimmed_mean" ,"centeredclipping", "signguard"]
            self.aggregators = [aggregator_registry[agg](config) for agg in agg_candidates]
            self.agg_candidates = agg_candidates
        else:
            self.hybrid = False
            self.aggregator = aggregator_registry[config.aggregator](config)
            self.aggregators = [aggregator_registry[agg](config) for agg, _ in aggregator_registry.items()]

        self.aggregator_names = list(aggregator_registry.keys())
        self.data_loader = kwargs["data_loader"]

        self.criterion = kwargs["criterion"]
        self.config = config
        self.random_agg = config.random_agg

    def global_step(self, model, benign_packages, attacker_packages, **kwargs):
        # merge benign and attacker packages, as we assume the server does not know which client is attacker
        benign_packages.update(attacker_packages)

        if self.random_agg:
            benign_packages.update(attacker_packages)
            idx = np.random.randint(0, len(self.agg_candidates))

            accumulated_delta = self.aggregators[idx](benign_packages)
            global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
            model.load_state_dict(global_weight.state_dict())

        elif self.hybrid:
            backup_weight = copy.deepcopy(model.state_dict())
            val_acc = []

            for i, agg in enumerate(self.aggregators):
                accumulated_delta = agg(benign_packages)
                global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
                model.load_state_dict(global_weight.state_dict())
            
                # validate each aggregation result
                acc, loss = test(self.data_loader, model, self.criterion, self.config)
                val_acc.append(acc)

                model.load_state_dict(backup_weight)
                # print("agg {:s} {:.4f}".format(self.aggregator_names[i], acc))

            max_acc_idx = np.argmax(val_acc)
            kwargs["logger"].info("max acc: {:s} {:.4f}".format(self.agg_candidates[max_acc_idx], val_acc[max_acc_idx]))

            accumulated_delta = self.aggregators[max_acc_idx](benign_packages)
            global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
            model.load_state_dict(global_weight.state_dict())

        else:
            benign_packages.update(attacker_packages)

            accumulated_delta = self.aggregator(benign_packages)
            global_weight = WeightBuffer(model.state_dict()) - accumulated_delta
            model.load_state_dict(global_weight.state_dict())


    @property
    def momentum(self):
        # return self.aggregator.momentum
        return self.aggregators[-2].momentum
