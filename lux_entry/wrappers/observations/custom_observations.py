from dataclasses import dataclass, asdict
import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, get_type_hints

from luxai_s2.state.state import ObservationStateDict, Team

from lux_entry.lux.state import Player


@dataclass
class FullObservation:
    """
    per map-tile features
    """

    # binary yes/no
    tile_has_ice: np.ndarray
    tile_has_ore: np.ndarray
    tile_has_lichen_strain: np.ndarray
    tile_per_player_has_factory: np.ndarray
    tile_per_player_has_robot: np.ndarray
    tile_per_player_has_light_robot: np.ndarray
    tile_per_player_has_heavy_robot: np.ndarray
    tile_per_player_has_lichen: np.ndarray
    # normalized from 0-1, -1 means inapplicable
    tile_rubble: np.ndarray
    tile_per_player_lichen: np.ndarray
    tile_per_player_light_robot_power: np.ndarray
    tile_per_player_light_robot_ice: np.ndarray
    tile_per_player_light_robot_ore: np.ndarray
    tile_per_player_heavy_robot_power: np.ndarray
    tile_per_player_heavy_robot_ice: np.ndarray
    tile_per_player_heavy_robot_ore: np.ndarray
    # normalized and positive unbounded, -1 means inapplicable
    tile_per_player_factory_ice_unbounded: np.ndarray
    tile_per_player_factory_ore_unbounded: np.ndarray
    tile_per_player_factory_water_unbounded: np.ndarray
    tile_per_player_factory_metal_unbounded: np.ndarray
    tile_per_player_factory_power_unbounded: np.ndarray
    """
    broadcast features
    """
    # normalized and positive unbounded
    total_per_player_robots_unbounded: np.ndarray
    total_per_player_light_robots_unbounded: np.ndarray
    total_per_player_heavy_robots_unbounded: np.ndarray
    total_per_player_factories_unbounded: np.ndarray
    total_per_player_factory_ice_unbounded: np.ndarray
    total_per_player_factory_ore_unbounded: np.ndarray
    total_per_player_factory_water_unbounded: np.ndarray
    total_per_player_factory_metal_unbounded: np.ndarray
    total_per_player_factory_power_unbounded: np.ndarray
    total_per_player_lichen_unbounded: np.ndarray
    # normalized from 0-1
    game_is_day: np.ndarray
    game_day_or_night_elapsed: np.ndarray
    game_time_elapsed: np.ndarray
    """
    bidding and factory placement info
    """
    teams: Dict[str, Team]  # not broadcast
    factories_per_team: int  # not broadcast
    valid_spawns_mask: np.ndarray


@dataclass
class Observation(FullObservation):
    "A unit's partial observation, drawn from FullObservation and centered around a unit's position."


def partial_obs_from(full_obs: FullObservation, pos: Tuple[int, int]) -> Observation:
    assert full_obs.tile_has_ice.shape == (
        1,
        48,
        48,
    )  # hopefully they don't change the map size
    x, y = 47 - pos[0], 47 - pos[1]

    def mean_pool(arr: np.ndarray, window: int) -> np.ndarray:
        arr = arr.reshape(
            arr.shape[0] // window, window, arr.shape[1] // window, window
        )
        return np.mean(arr, axis=(1, 3))

    obs = dict()
    for key, value in asdict(full_obs).items():
        if key in ["teams", "factories_per_team"]:
            obs[key] = value
            continue
        expanded_map = np.full((value.shape[0], 96, 96), -1.0)
        obs[key] = np.zeros((value.shape[0] * 4, 12, 12))
        for p in range(value.shape[0]):
            expanded_map[p][x : x + 48, y : y + 48] = value[
                p
            ]  # unit is in lower right pixel of upper left quadrant
            obs[key][p * 4] = expanded_map[p][42:54, 42:54]  # min map (12x12 area)
            obs[key][p * 4 + 1] = mean_pool(
                expanded_map[p][36:60, 36:60], 2
            )  # med map (24x24 area)
            obs[key][p * 4 + 2] = mean_pool(
                expanded_map[p][24:72, 24:72], 4
            )  # max map (48x48 area)
            obs[key][p * 4 + 3] = mean_pool(expanded_map[p], 8)  # max map (96x96 area)
    return Observation(**obs)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env_cfg = self.env.state.env_cfg
        map_size = self.env.state.env_cfg.map_size

        spaces_dict = spaces.Dict()
        for key in get_type_hints(FullObservation).keys():
            n_features = (
                2
                if "_per_player_" in key
                else 2 * self.env_cfg.MAX_FACTORIES
                if key == "tile_has_lichen_strain"
                else 1
            )
            low = (
                -1.0
                if (("_robot_" in key or "_factory_" in key) and "total_" not in key)
                else 0.0
            )
            high = np.inf if "_unbounded" in key else 1.0
            spaces_dict[key] = (
                spaces.MultiBinary((1, n_features, map_size, map_size))
                if "_has_" in key or "_is_" in key
                else spaces.Dict()
                if key == "teams"
                else int
                if key == "factories_per_team"
                else spaces.Box(low, high, shape=(1, n_features, map_size, map_size))
            )
        assert spaces_dict.keys() == get_type_hints(FullObservation).keys()
        self.observation_space = spaces_dict

    def observation(
        self, env_obs_both_players: Dict[Player, ObservationStateDict]
    ) -> FullObservation:
        MAX_FS = self.env_cfg.MAX_FACTORIES
        MAX_RUBBLE = self.env_cfg.MAX_RUBBLE
        MAX_LICHEN = self.env_cfg.MAX_LICHEN_PER_TILE
        LIGHT_CARGO_SPACE = self.env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
        LIGHT_BAT_CAP = self.env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY
        HEAVY_CARGO_SPACE = self.env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
        HEAVY_BAT_CAP = self.env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY
        EXP_MAX_F_CARGO = 2 * HEAVY_CARGO_SPACE
        ICE_WATER_RATIO = self.env_cfg.ICE_WATER_RATIO
        ORE_METAL_RATIO = self.env_cfg.ORE_METAL_RATIO
        EXP_MAX_F_WATER = EXP_MAX_F_CARGO / ICE_WATER_RATIO
        EXP_MAX_F_METAL = EXP_MAX_F_CARGO / ORE_METAL_RATIO
        EXP_MAX_TOT_F_CARGO = MAX_FS * EXP_MAX_F_CARGO
        EXP_MAX_TOT_F_WATER = EXP_MAX_TOT_F_CARGO / ICE_WATER_RATIO
        EXP_MAX_TOT_F_METAL = EXP_MAX_TOT_F_CARGO / ORE_METAL_RATIO
        EXP_MAX_F_POWER = self.env_cfg.FACTORY_CHARGE * 30
        EXP_MAX_TOT_F_POWER = EXP_MAX_F_POWER * MAX_FS
        EXP_MAX_RS = 50
        CYCLE_LENGTH = self.env_cfg.CYCLE_LENGTH
        DAY_LENGTH = self.env_cfg.DAY_LENGTH
        MAX_EPISODE_LENGTH = self.env_cfg.max_episode_length

        env_obs = env_obs_both_players[
            "player_0"
        ]  # env_obs["player_0"] == env_obs["player_1"]
        obs = dict()
        for key, value in self.observation_space.items():
            if key in ["teams", "factories_per_team"]:
                continue
            obs[key] = (
                np.zeros(value.shape[1:])
                if type(value) == spaces.MultiBinary
                else np.zeros(value.shape[1:]) + value.low[0]
            )
        obs["tile_has_ice"][0] = env_obs["board"]["ice"]
        obs["tile_has_ore"][0] = env_obs["board"]["ore"]
        for i in range(2 * self.env_cfg.MAX_FACTORIES):
            obs["tile_has_lichen_strain"][i] = env_obs["board"]["lichen_strains"] == i
        obs["tile_rubble"][0] = env_obs["board"]["rubble"] / MAX_RUBBLE
        lichen_strains = [[], []]
        for p, player in enumerate(["player_0", "player_1"]):
            for f in env_obs["factories"][player].values():
                cargo = f["cargo"]
                pos = (p, f["pos"][0], f["pos"][1])
                lichen_strains[p].append(f["strain_id"])
                obs["tile_per_player_has_factory"][pos] = 1
                obs["tile_per_player_factory_ice_unbounded"][pos] = (
                    cargo["ice"] / EXP_MAX_F_CARGO
                )
                obs["tile_per_player_factory_ore_unbounded"][pos] = (
                    cargo["ore"] / EXP_MAX_F_CARGO
                )
                obs["tile_per_player_factory_water_unbounded"][pos] = (
                    cargo["water"] / EXP_MAX_F_WATER
                )
                obs["tile_per_player_factory_metal_unbounded"][pos] = (
                    cargo["metal"] / EXP_MAX_F_METAL
                )
                obs["tile_per_player_factory_power_unbounded"][pos] = (
                    f["power"] / EXP_MAX_F_POWER
                )
                obs["total_per_player_factory_ice_unbounded"][p] += (
                    cargo["ice"] / EXP_MAX_TOT_F_CARGO
                )
                obs["total_per_player_factory_ore_unbounded"][p] += (
                    cargo["ore"] / EXP_MAX_TOT_F_CARGO
                )
                obs["total_per_player_factory_water_unbounded"][p] += (
                    cargo["water"] / EXP_MAX_TOT_F_WATER
                )
                obs["total_per_player_factory_metal_unbounded"][p] += (
                    cargo["metal"] / EXP_MAX_TOT_F_METAL
                )
                obs["total_per_player_factory_power_unbounded"][p] += (
                    f["power"] / EXP_MAX_TOT_F_POWER
                )
            for r in env_obs["units"][player].values():
                cargo = r["cargo"]
                pos = (p, r["pos"][0], r["pos"][1])
                obs["tile_per_player_has_robot"][pos] = 1
                obs["tile_has_light_robot"][pos] = r["unit_type"] == "LIGHT"
                obs["tile_has_heavy_robot"][pos] = r["unit_type"] == "HEAVY"
                obs["tile_per_player_light_robot_power"][pos] = (
                    r["power"] / LIGHT_BAT_CAP
                )
                obs["tile_per_player_light_robot_ice"][pos] = (
                    cargo["ice"] / LIGHT_CARGO_SPACE
                )
                obs["tile_per_player_light_robot_ore"][pos] = (
                    cargo["ore"] / LIGHT_CARGO_SPACE
                )
                obs["tile_per_player_heavy_robot_power"][pos] = (
                    r["power"] / HEAVY_BAT_CAP
                )
                obs["tile_per_player_heavy_robot_ice"][pos] = (
                    cargo["ice"] / HEAVY_CARGO_SPACE
                )
                obs["tile_per_player_heavy_robot_ore"][pos] = (
                    cargo["ore"] / HEAVY_CARGO_SPACE
                )
                obs["total_per_player_light_robots_unbounded"][p] += (
                    1 / EXP_MAX_RS * (r["unit_type"] == "LIGHT")
                )
                obs["total_per_player_heavy_robots_unbounded"][p] += (
                    1 / EXP_MAX_RS * (r["unit_type"] == "HEAVY")
                )
            obs["total_per_player_robots_unbounded"][p] += (
                len(env_obs["units"][player]) / EXP_MAX_RS
            )
            obs["tile_per_player_has_lichen"][p] = (
                env_obs["board"]["lichen_strains"] == lichen_strains[p]
            )
            obs["tile_per_player_lichen"][p] = (
                env_obs["board"]["lichen"]
                / MAX_LICHEN
                * (env_obs["board"]["lichen_strains"] == lichen_strains[p])
            )
            obs["total_per_player_lichen_unbounded"][p] = np.sum(
                obs["tile_per_player_lichen"][p]
            )
        game_is_day = env_obs["real_env_steps"] % CYCLE_LENGTH < DAY_LENGTH
        obs["game_is_day"][0] += game_is_day
        obs["game_day_or_night_elapsed"][0] += (
            (env_obs["real_env_steps"] % CYCLE_LENGTH) / DAY_LENGTH
            if game_is_day
            else (env_obs["real_env_steps"] % CYCLE_LENGTH - DAY_LENGTH)
            / (CYCLE_LENGTH - DAY_LENGTH)
        )
        obs["game_time_elapsed"][0] += env_obs["real_env_steps"] / MAX_EPISODE_LENGTH
        obs["teams"] = env_obs["teams"]
        obs["factories_per_team"] = env_obs["board"]["factories_per_team"]
        obs["valid_spawns_mask"][0] = env_obs["board"]["valid_spawns_mask"]

        return FullObservation(**obs)
