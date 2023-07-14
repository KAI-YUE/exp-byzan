from fedlearning.buffer import WeightBuffer

class _BaseAggregator(object):
    """Base class of aggregators.

    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, config, **kwargs):
        self.store_momentum = config.store_momentum
        self.momentum = None

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        """
        self._pre_aggregate(inputs)
        accumulated_delta = self._aggregate(inputs)
        self._post_aggregate(accumulated_delta)

        return accumulated_delta

    def _pre_aggregate(self, local_packages, **kwargs):
        # if momentum is None, initialize 
        if self.store_momentum and self.momentum is None:
            self.momentum = WeightBuffer(local_packages[0].state_dict())

    def _post_aggregate(self, accumulated_delta, **kwargs):
        if self.store_momentum:
            self.momentum = accumulated_delta