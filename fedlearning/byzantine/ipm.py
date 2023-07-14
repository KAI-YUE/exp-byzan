from fedlearning.client import Client

class IPMAttacker(Client):
    def __init__(self, config, model, **kwargs):
        super(IPMAttacker, self).__init__(config, model, **kwargs)
        self.ipm_multiplier = config.ipm_multiplier
    
    def init_local_dataset(self, *args):
        pass

    def local_step(self, oracle, **kwargs):
        self.delta = oracle * (-self.ipm_multiplier)

    def compute_delta(self):
        return self.delta