from typing import Generator

import numpy as np
import torch

from fedlearning.client import Client
from fedlearning.aggregators.krum import Multikrum
from fedlearning.buffer import _get_para

class FangattackAdversary(Client):
    r""""""

    def __init__(self, config, model):
        super().__init__(config, model)

    def attack_median_and_trimmedmean(self, benign_packages):

        benign_update = self._get_updates(benign_packages)
        agg_grads = torch.mean(benign_update, 0)

        deviation = torch.sign(agg_grads)
        device = benign_update.device
        b = 2
        max_vector = torch.max(benign_update, 0)[0]
        min_vector = torch.min(benign_update, 0)[0]

        max_ = (max_vector > 0).type(torch.FloatTensor).to(device)
        min_ = (min_vector < 0).type(torch.FloatTensor).to(device)

        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b

        max_range = torch.cat(
            (max_vector[:, None], (max_vector * max_)[:, None]), dim=1
        )
        min_range = torch.cat(
            ((min_vector * min_)[:, None], min_vector[:, None]), dim=1
        )

        rand = (
            torch.from_numpy(
                np.random.uniform(0, 1, [len(deviation), self.num_byzantine])
            )
            .type(torch.FloatTensor)
            .to(benign_update.device)
        )

        max_rand = (
            torch.stack([max_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        )
        min_rand = (
            torch.stack([min_range[:, 0]] * rand.shape[1]).T
            + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
        )

        mal_vec = (
            torch.stack(
                [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]
            ).T.to(device)
            * max_rand
            + torch.stack(
                [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]
            ).T.to(device)
            * min_rand
        ).T

        shape, num_params = self._retrieve_shape(benign_packages[0].state_dict())
        delta = self._updates_to_state_dict(mal_vec, shape, num_params)

        return delta

    def attack_multikrum(self, benign_packages, local_packages):
        multi_krum = Multikrum(self.config)

        benign_update = self._get_updates(benign_packages)
        agg_updates = torch.mean(benign_update, 0)
        all_updates = self._get_updates(local_packages) 
        deviation = torch.sign(agg_updates)

        def compute_lambda(all_updates, model_re, n_attackers):

            distances = []
            n_benign, d = all_updates.shape
            for update in all_updates:
                distance = torch.norm((all_updates - update), dim=1)
                distances = (
                    distance[None, :]
                    if not len(distances)
                    else torch.cat((distances, distance[None, :]), 0)
                )

            distances[distances == 0] = 10000
            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(
                distances[:, : n_benign - 2 - n_attackers],
                dim=1,
            )
            min_score = torch.min(scores)
            term_1 = min_score / (
                (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
            )
            max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (
                torch.sqrt(torch.Tensor([d]))[0]
            )

            return term_1 + max_wre_dist

        lambda_ = compute_lambda(all_updates, agg_updates, self.num_byzantine)

        threshold = 1e-5
        mal_update = []

        while lambda_ > threshold:
            mal_update = -lambda_ * deviation
            mal_updates = torch.stack([mal_update] * self.num_byzantine)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            # print(mal_updates.shape, n_attackers)
            agg_grads, krum_candidate = multi_krum(mal_updates)
            if krum_candidate < self.num_byzantine:
                return mal_update
            else:
                mal_update = []

            lambda_ *= 0.5

        if not len(mal_update):
            mal_update = agg_updates - lambda_ * deviation

        shape, num_params = self._retrieve_shape(local_packages[0].state_dict())
        delta = torch.zeros(num_params)


    def local_step(self, criterion, **kwargs):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model
        """
        self.benign_packages = kwargs["benign_packages"]

    def uplink_transmit(self):
        
        delta = self.attack_median_and_trimmedmean(self.benign_packages)
        return self.delta


    def _get_updates(self, local_packages):
        updates = []
        for i, user_idx in enumerate(local_packages):
            state_dict = local_packages[user_idx].state_dict()
            updates.append(_get_para(state_dict))
            
        updates = torch.stack(updates)
        return updates
    
    def _updates_to_state_dict(self, updates, shape, num_params):
        state_dict = {}
        num_params.append(-1)
        p, q = 0, num_params[0]
        for i, key in enumerate(shape):
            state_dict[key] = updates[p:q].reshape(shape[key])
            p = q
            q += num_params[i+1]

        return state_dict
