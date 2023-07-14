from fedlearning.aggregators.initialize import _BaseAggregator

class Vote(_BaseAggregator):
    r"""Computes the vote results of signs."""
    def __init__(self, config):
        super(Vote, self).__init__(config)
        self.num_users = config.total_users
        self.lr = config.global_lr

    def __call__(self, local_packages):
        accumulated_delta = local_packages[0]
        for user_id, package in local_packages.items():
            if user_id == 0:
                continue
            accumulated_delta = accumulated_delta + package

        # take the sign and multiply by the learning rate
        for w_weight, delta in accumulated_delta._weight_dict.items():
            accumulated_delta._weight_dict[w_weight] = delta.sign() * self.lr

        return accumulated_delta


