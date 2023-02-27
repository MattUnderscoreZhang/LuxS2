import numpy as np
import torch
from typing import Tuple


def convert_obs_to_tensor(all_observables: list[np.ndarray], pass_through_observables: list[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    for obs in all_observables + pass_through_observables:
        assert (
            len(obs.shape) == 4
            and obs.shape[0] == 1
            and obs.shape[2] == 12
            and obs.shape[3] == 12
        )
    all_obs = torch.cat([torch.from_numpy(obs) for obs in all_observables], dim=1)
    pass_obs = torch.cat([torch.from_numpy(obs) for obs in pass_through_observables], dim=1)
    return all_obs, pass_obs
