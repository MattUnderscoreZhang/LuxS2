import numpy as np
import torch
from typing import Tuple


def construct_obs(conv_obs: list[np.ndarray], skip_obs: list[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    for obs in conv_obs + skip_obs:
        assert (
            len(obs.shape) == 4
            and obs.shape[0] == 1
            # variable second dimension
            and obs.shape[2] == 12
            and obs.shape[3] == 12
        )
    return(
        torch.cat([torch.from_numpy(obs) for obs in conv_obs], dim=1),
        torch.cat([torch.from_numpy(obs) for obs in skip_obs], dim=1)
    )
