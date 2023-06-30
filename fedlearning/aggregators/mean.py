from fedlearning.aggregators.initialize import _BaseAggregator

class Mean(_BaseAggregator):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config):
        self.num_users = config.total_users

    def __call__(self, local_packages):
        accumulated_delta = local_packages[0]
        for user_id, package in local_packages.items():
            if user_id == 0:
                continue
            accumulated_delta = accumulated_delta + package

        accumulated_delta = accumulated_delta * (1/self.num_users)

        return accumulated_delta


class BenignFedAvg(Mean):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, num_users):
        self.num_users = num_users
