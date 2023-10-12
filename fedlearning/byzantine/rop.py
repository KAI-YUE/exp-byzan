import numpy as np
import torch

from fedlearning.client import Client
from fedlearning.buffer import WeightBuffer

class RopAttacker(Client):
    r"""Computes the ``sample mean`` over the updates from all give clients."""
    def __init__(self, config, model, **kwargs):
        super(RopAttacker, self).__init__(config, model, **kwargs)
        self.num_users = config.total_users
        self.PI = config.PI/180 * np.pi # convert degrees to radians
        self.scaling_factor = config.rop_scaling_factor

    def local_step(self, momentum=None, **kwargs):
        if momentum is None:
            self.delta = self.w0
            return None
        
        delta = {}
        for w_name, m in momentum._weight_dict.items():
            m_ = m.view(-1)
            p = torch.ones_like(m_)
            p_tilde = (p @ m_)/torch.norm(m_)**2 * m_
            orthogonal = p - p_tilde
            tmp = np.sin(self.PI)*orthogonal/torch.norm(orthogonal) + np.cos(self.PI)*m_/torch.norm(m_)
            delta[w_name] = tmp.view(m.shape)

        self.delta = WeightBuffer(delta)
        self.complete_attack = True

    def compute_delta(self):
        delta = self.delta * self.scaling_factor
        return delta
