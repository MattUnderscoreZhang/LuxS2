import gym
from typing import Dict

from luxai_s2.state.state import ObservationStateDict

from lux_entry.components.map_features_obs import MapFeaturesObservation, get_full_obs_space, get_full_two_player_obs
from lux_entry.lux.state import Player


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        map_size = self.env.state.env_cfg.map_size
        self.observation_space = get_full_obs_space(self.env_cfg, map_size)

    def observation(self, two_player_env_obs: Dict[Player, ObservationStateDict]) -> Dict[Player, MapFeaturesObservation]:
        return get_full_two_player_obs(two_player_env_obs, self.env_cfg, self.observation_space)
