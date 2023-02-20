from lux.config import EnvConfig
import torch

from . import model, training_env
from lux_entry.behaviors import load_model
from lux_entry.lux.state import Player


class Agent:
    def __init__(self, player: Player, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = load_model(model.Net, model.WEIGHTS_PATH)
        self.policy.eval().to(device)

        self.controller = training_env.ControllerWrapper(self.env_cfg)

    def bid_policy(self, step: int, obs: dict, remainingOverageTime: int = 60):
        return training_env.bid_policy(player=self.player, obs=obs)

    def factory_placement_policy(self, step: int, obs: dict, remainingOverageTime: int = 60):
        return training_env.factory_placement_policy(player=self.player, obs=obs)

    def act(self, step: int, obs: dict, remainingOverageTime: int = 60):
        raw_obs = {
            "player_0": obs,
            "player_1": obs,
        }
        obs = training_env.ObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs_arr = obs[self.player]

        obs_arr = torch.from_numpy(obs_arr).float()
        with torch.no_grad():
            action_mask = (
                torch.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)  # we unsqueeze/add an extra batch dimension =
                .bool()
            )
            actions = (
                self.policy.act(
                    obs_arr.unsqueeze(0), deterministic=False, action_masks=action_mask
                )
                .cpu()
                .numpy()
            )
        return self.controller.action_to_lux_action(self.player, raw_obs, actions[0])
