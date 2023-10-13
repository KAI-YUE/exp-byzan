from typing import List
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import inf

from fedlearning.aggregators.initialize import _BaseAggregator
from fedlearning.buffer import _get_para, WeightBuffer

class SignGuard(_BaseAggregator):
    r"""A robust aggregator from paper `Xu et al.

    SignGuard: Byzantine-robust Federated
    Learning through Collaborative Malicious Gradient
    Filtering <https://arxiv.org/abs/2109.05872>`_.
    """

    def __init__(self, config):
        super(SignGuard, self).__init__(config)

        self.l2norm_his = []
        if config.agg == "mean":
            self.agg = torch.mean
        elif config.agg == "median":
            self.agg = torch.median

    def _get_updates(self, local_packages):
        updates = []
        for i, user_idx in enumerate(local_packages):
            state_dict = local_packages[user_idx].state_dict()
            updates.append(_get_para(state_dict))
            
        updates = torch.stack(updates)
        return updates

    def _aggregate(self, local_packages):
        updates = self._get_updates(local_packages)
        num = len(updates)
        l2norms = [torch.norm(update).item() for update in updates]
        M = np.median(l2norms)

        for idx in range(num):
            if l2norms[idx] > M:
                updates[idx] = clip_tensor_norm_(updates[idx], M)
                l2norms[idx] = torch.norm(updates[idx]).item()
        L = 0.1
        R = 3.0
        S1_idxs = []
        for idx, (l2norm, update) in enumerate(zip(l2norms, updates)):
            if l2norm >= L * M and l2norm <= R * M:
                S1_idxs.append(idx)

        features = []
        num_para = len(updates[0])
        for update in updates:
            feature0 = (update > 0).sum().item() / num_para
            feature1 = (update < 0).sum().item() / num_para
            feature2 = (update == 0).sum().item() / num_para

            features.append([feature0, feature1, feature2])

        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(features)

        flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
        S2_idxs = list(
            [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
        )

        inter = list(set(S1_idxs) & set(S2_idxs))
        benign_updates = []
        for idx in inter:
            if l2norms[idx] > M:
                updates[idx] = clip_tensor_norm_(updates[idx], M)
            benign_updates.append(updates[idx])

        benign_updates = torch.stack(benign_updates)
        updates = self.agg(benign_updates, dim=0)
        
        shape, num_params = self._retrieve_shape(local_packages[0].state_dict())
        delta = self._updates_to_state_dict(updates, shape, num_params)
        
        return WeightBuffer(delta)

    def _retrieve_shape(self, state_dict):
        shape = {}
        num_params = []
        for i, key in enumerate(state_dict):
            shape[key] = state_dict[key].shape
            num_params.append(state_dict[key].numel())

        return shape, num_params
    
    def _updates_to_state_dict(self, updates, shape, num_params):
        state_dict = {}
        num_params.append(-1)
        p, q = 0, num_params[0]
        for i, key in enumerate(shape):
            state_dict[key] = updates[p:q].reshape(shape[key])
            p = q
            q += num_params[i+1]

        return state_dict


def clip_tensor_norm_(
    parameters, max_norm: float, norm_type=2, error_if_nonfinite=True):

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.detach(), norm_type).to(device)
                    for p in parameters
                    if p.dtype != torch.int64
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1,
    # but doing so avoids a `if clip_coef < 1:` conditional which can require a
    # CPU <=> device synchronization when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            return p.detach().mul_(clip_coef_clamped.to(p.device))