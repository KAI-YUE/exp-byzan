class _BaseAggregator(object):
    """Base class of aggregators.

    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, *args, **kwargs):
        pass
        # log("Init aggregators: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.

        Args:
            inputs (list): A list of tensors to be aggregated.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError
