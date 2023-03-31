from dataclasses import dataclass
from gym import spaces
import numpy as np
import torch
from typing import get_type_hints

from luxai_s2.state.state import ObservationStateDict, Team

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player


@dataclass
class MapFeaturesObservation:
    """
    per map-tile features
    """
    # binary yes/no
    has_ice: np.ndarray
    has_ore: np.ndarray
    has_lichen_strain: np.ndarray
    player_has_factory: np.ndarray
    player_has_robot: np.ndarray
    player_has_light_robot: np.ndarray
    player_has_heavy_robot: np.ndarray
    player_has_lichen: np.ndarray
    # normalized from 0-1, -1 means inapplicable
    rubble: np.ndarray
    player_lichen: np.ndarray
    player_light_robot_power: np.ndarray
    player_light_robot_ice: np.ndarray
    player_light_robot_ore: np.ndarray
    player_heavy_robot_power: np.ndarray
    player_heavy_robot_ice: np.ndarray
    player_heavy_robot_ore: np.ndarray
    # normalized and positive unb, -1 means inapplicable
    player_factory_ice_unb: np.ndarray
    player_factory_ore_unb: np.ndarray
    player_factory_water_unb: np.ndarray
    player_factory_metal_unb: np.ndarray
    player_factory_power_unb: np.ndarray
    """
    broadcast features
    """
    # normalized and positive unb
    player_tot_robots_unb: np.ndarray
    player_tot_light_robots_unb: np.ndarray
    player_tot_heavy_robots_unb: np.ndarray
    player_tot_factories_unb: np.ndarray
    player_tot_factory_ice_unb: np.ndarray
    player_tot_factory_ore_unb: np.ndarray
    player_tot_factory_water_unb: np.ndarray
    player_tot_factory_metal_unb: np.ndarray
    player_tot_factory_power_unb: np.ndarray
    player_tot_lichen_unb: np.ndarray
    # normalized from 0-1
    game_is_day: np.ndarray
    game_day_or_night_elapsed: np.ndarray
    game_time_elapsed: np.ndarray
    """
    bidding and factory placement info
    """
    teams: dict[str, Team]  # not broadcast
    factories_per_team: int  # not broadcast
    valid_spawns_mask: np.ndarray


def get_full_obs_space(env_cfg: EnvConfig) -> spaces.Dict:
    map_size = env_cfg.map_size
    spaces_dict = spaces.Dict()
    for key in get_type_hints(MapFeaturesObservation).keys():
        n_features = (
            2
            if "player_" in key
            else 2 * env_cfg.MAX_FACTORIES
            if key == "has_lichen_strain"
            else 1
        )
        low = (
            -1.0
            if (("_robot_" in key or "_factory_" in key) and "total_" not in key)
            else 0.0
        )
        high = np.inf if "_unb" in key else 1.0
        spaces_dict[key] = (
            spaces.MultiBinary((n_features, map_size, map_size))
            if "has_" in key or "is_" in key
            else spaces.Dict()
            if key == "teams"
            else int
            if key == "factories_per_team"
            else spaces.Box(low, high, shape=(n_features, map_size, map_size))
        )
    assert spaces_dict.keys() == get_type_hints(MapFeaturesObservation).keys()
    return spaces_dict


def get_full_obs(
    env_obs: ObservationStateDict, env_cfg: EnvConfig, player: Player, opponent: Player,
) -> MapFeaturesObservation:
    # normalization factors
    MAX_FS = env_cfg.MAX_FACTORIES
    MAX_RUBBLE = env_cfg.MAX_RUBBLE
    MAX_LICHEN = env_cfg.MAX_LICHEN_PER_TILE
    LIGHT_CARGO_SPACE = env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
    LIGHT_BAT_CAP = env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY
    HEAVY_CARGO_SPACE = env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
    HEAVY_BAT_CAP = env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY
    EXP_MAX_F_CARGO = 2 * HEAVY_CARGO_SPACE
    ICE_WATER_RATIO = env_cfg.ICE_WATER_RATIO
    ORE_METAL_RATIO = env_cfg.ORE_METAL_RATIO
    EXP_MAX_F_WATER = EXP_MAX_F_CARGO / ICE_WATER_RATIO
    EXP_MAX_F_METAL = EXP_MAX_F_CARGO / ORE_METAL_RATIO
    EXP_MAX_TOT_F_CARGO = MAX_FS * EXP_MAX_F_CARGO
    EXP_MAX_TOT_F_WATER = EXP_MAX_TOT_F_CARGO / ICE_WATER_RATIO
    EXP_MAX_TOT_F_METAL = EXP_MAX_TOT_F_CARGO / ORE_METAL_RATIO
    EXP_MAX_F_POWER = env_cfg.FACTORY_CHARGE * 30
    EXP_MAX_TOT_F_POWER = EXP_MAX_F_POWER * MAX_FS
    EXP_MAX_RS = 50
    CYCLE_LENGTH = env_cfg.CYCLE_LENGTH
    DAY_LENGTH = env_cfg.DAY_LENGTH
    MAX_EPISODE_LENGTH = env_cfg.max_episode_length

    # init keys in observation as array of zeros of the right size
    observation_space = get_full_obs_space(env_cfg)
    obs = {
        key: np.zeros(value.shape)
        if type(value) == spaces.MultiBinary
        else np.zeros(value.shape) + value.low[0]
        for key, value in observation_space.items()
        if key not in ["teams", "factories_per_team"]
    }

    # fill in observations
    obs["has_ice"][0] = env_obs["board"]["ice"]
    obs["has_ore"][0] = env_obs["board"]["ore"]
    for i in range(2 * env_cfg.MAX_FACTORIES):
        obs["has_lichen_strain"][i] = env_obs["board"]["lichen_strains"] == i
    obs["rubble"][0] = env_obs["board"]["rubble"] / MAX_RUBBLE
    lichen_strains = [[], []]
    for p, player in enumerate([player, opponent]):
        for f in env_obs["factories"][player].values():
            cargo = f["cargo"]
            pos = (p, f["pos"][0], f["pos"][1])
            lichen_strains[p].append(f["strain_id"])
            obs["player_has_factory"][pos] = 1
            obs["player_factory_ice_unb"][pos] = cargo["ice"] / EXP_MAX_F_CARGO
            obs["player_factory_ore_unb"][pos] = cargo["ore"] / EXP_MAX_F_CARGO
            obs["player_factory_water_unb"][pos] = cargo["water"] / EXP_MAX_F_WATER
            obs["player_factory_metal_unb"][pos] = cargo["metal"] / EXP_MAX_F_METAL
            obs["player_factory_power_unb"][pos] = f["power"] / EXP_MAX_F_POWER
            obs["player_tot_factory_ice_unb"][p] += cargo["ice"] / EXP_MAX_TOT_F_CARGO
            obs["player_tot_factory_ore_unb"][p] += cargo["ore"] / EXP_MAX_TOT_F_CARGO
            obs["player_tot_factory_water_unb"][p] += cargo["water"] / EXP_MAX_TOT_F_WATER
            obs["player_tot_factory_metal_unb"][p] += cargo["metal"] / EXP_MAX_TOT_F_METAL
            obs["player_tot_factory_power_unb"][p] += f["power"] / EXP_MAX_TOT_F_POWER
        for r in env_obs["units"][player].values():
            cargo = r["cargo"]
            pos = (p, r["pos"][0], r["pos"][1])
            obs["player_has_robot"][pos] = 1
            obs["player_has_light_robot"][pos] = r["unit_type"] == "LIGHT"
            obs["player_has_heavy_robot"][pos] = r["unit_type"] == "HEAVY"
            obs["player_light_robot_power"][pos] = r["power"] / LIGHT_BAT_CAP
            obs["player_light_robot_ice"][pos] = cargo["ice"] / LIGHT_CARGO_SPACE
            obs["player_light_robot_ore"][pos] = cargo["ore"] / LIGHT_CARGO_SPACE
            obs["player_heavy_robot_power"][pos] = r["power"] / HEAVY_BAT_CAP
            obs["player_heavy_robot_ice"][pos] = cargo["ice"] / HEAVY_CARGO_SPACE
            obs["player_heavy_robot_ore"][pos] = cargo["ore"] / HEAVY_CARGO_SPACE
            obs["player_tot_light_robots_unb"][p] += 1 / EXP_MAX_RS * (r["unit_type"] == "LIGHT")
            obs["player_tot_heavy_robots_unb"][p] += 1 / EXP_MAX_RS * (r["unit_type"] == "HEAVY")
        obs["player_tot_robots_unb"][p] += len(env_obs["units"][player]) / EXP_MAX_RS
        obs["player_has_lichen"][p] = env_obs["board"]["lichen_strains"] == lichen_strains[p]
        obs["player_lichen"][p] = (
            env_obs["board"]["lichen"]
            / MAX_LICHEN
            * (env_obs["board"]["lichen_strains"] == lichen_strains[p])
        )
        obs["player_tot_lichen_unb"][p] = np.sum(obs["player_lichen"][p])
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

    return MapFeaturesObservation(**obs)


def get_minimap_obs(
    full_conv_obs: list[np.ndarray], full_skip_obs: list[np.ndarray], pos: np.ndarray
) -> dict[str, torch.Tensor]:
    """
    Create minimaps for a set of features around (x, y).
    Conv obs are ones which will be convoluted together to extract high-level features.
    Skip obs are used directly by the agent without alteration.
    """
    def _mean_pool(arr: np.ndarray, window: int) -> np.ndarray:
        arr = arr.reshape(
            arr.shape[0] // window, window, arr.shape[1] // window, window
        )
        return np.mean(arr, axis=(1, 3))

    def _get_minimap(value: np.ndarray, x: int, y: int) -> np.ndarray:
        expanded_map = np.full((value.shape[0], 96, 96), -1.0)
        minimap = np.zeros((value.shape[0] * 4, 12, 12))
        for p in range(value.shape[0]):
            # unit is in lower right pixel of upper left quadrant
            expanded_map[p][x : x + 48, y : y + 48] = value[p]
            # small map (12x12 area)
            minimap[p * 4] = expanded_map[p][42:54, 42:54]
            # medium map (24x24 area)
            minimap[p * 4 + 1] = _mean_pool(expanded_map[p][36:60, 36:60], 2)
            # large map (48x48 area)
            minimap[p * 4 + 2] = _mean_pool(expanded_map[p][24:72, 24:72], 4)
            # full map (96x96 area)
            minimap[p * 4 + 3] = _mean_pool(expanded_map[p], 8)
        assert (
            len(minimap.shape) == 3
            # variable second dimension
            and minimap.shape[1] == 12
            and minimap.shape[2] == 12
        )
        return minimap

    conv_obs = [_get_minimap(value, pos[0], pos[1]) for value in full_conv_obs]
    skip_obs = [_get_minimap(value, pos[0], pos[1]) for value in full_skip_obs]

    return {
        "conv_obs": torch.cat([torch.from_numpy(obs) for obs in conv_obs], dim=0),
        "skip_obs": torch.cat([torch.from_numpy(obs) for obs in skip_obs], dim=0),
    }


jobs = [
    "ice_miner",
    "ore_miner",
    "courier",
    "sabateur",
    "scout",
    "soldier",
    "builder",
    "factory",
]

"""
units_obs = [
    full_obs.player_has_robot,
    full_obs.player_has_light_robot,
    full_obs.player_has_heavy_robot,
    full_obs.player_light_robot_power,
    full_obs.player_heavy_robot_power,
    full_obs.player_tot_robots_unb,
    full_obs.player_tot_light_robots_unb,
    full_obs.player_tot_heavy_robots_unb,
]
time_obs = [
    full_obs.game_is_day,
    full_obs.game_day_or_night_elapsed,
    full_obs.game_time_elapsed,
]
map_factory_obs = [
    full_obs.player_has_factory,
]
factory_ice_obs = [
    full_obs.player_factory_ice_unb,
    full_obs.player_factory_water_unb,
    full_obs.player_tot_factory_ice_unb,
    full_obs.player_tot_factory_water_unb,
]
factory_ore_obs = [
    full_obs.player_factory_ore_unb,
    full_obs.player_factory_metal_unb,
    full_obs.player_tot_factory_ore_unb,
    full_obs.player_tot_factory_metal_unb,
]
factory_power_obs = [
    full_obs.player_factory_power_unb,
    full_obs.player_tot_factory_power_unb,
]
cargo_obs = [
    full_obs.player_light_robot_ice,
    full_obs.player_light_robot_ore,
    full_obs.player_heavy_robot_ice,
    full_obs.player_heavy_robot_ore,
]
map_ice_obs = [
    full_obs.has_ice,
]
map_ore_obs = [
    full_obs.has_ore,
]
map_rubble_obs = [
    full_obs.rubble,
]
lichen_obs = [
    full_obs.has_lichen_strain,
    full_obs.player_has_lichen,
    full_obs.player_lichen,
    full_obs.player_tot_lichen_unb,
]
bidding_obs = [
    full_obs.teams,
    full_obs.factories_per_team,
    full_obs.player_tot_factories_unb,
    full_obs.valid_spawns_mask,
]
"""


def get_obs_by_job(
    full_obs: MapFeaturesObservation,
    job: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if job == "ice_miner":
        full_conv_obs = [
            # best factory to send ice
            full_obs.player_factory_ice_unb[0],
            full_obs.player_factory_water_unb[0],
            # power info
            np.add(full_obs.player_light_robot_power[0], full_obs.player_heavy_robot_power[0]),
            full_obs.player_factory_power_unb[0],
            full_obs.game_is_day,
            full_obs.game_day_or_night_elapsed,
            # other robots
            full_obs.player_has_robot,
            full_obs.player_has_light_robot[0],
            full_obs.player_has_heavy_robot[0],
            # navigation
            full_obs.rubble,
        ]
        full_skip_obs = [
            # where ice is
            full_obs.has_ice,
            # cargo status
            np.add(full_obs.player_light_robot_ice[0], full_obs.player_heavy_robot_ice[0]),
            # where factories are
            full_obs.player_has_factory[0],
        ]
    elif job == "ore_miner":
        full_conv_obs = [
            # best factory to send ore
            full_obs.player_factory_ore_unb[0],
            full_obs.player_factory_metal_unb[0],
            # power info
            np.add(full_obs.player_light_robot_power[0], full_obs.player_heavy_robot_power[0]),
            full_obs.player_factory_power_unb[0],
            full_obs.game_is_day,
            full_obs.game_day_or_night_elapsed,
            # other robots
            full_obs.player_has_robot,
            full_obs.player_has_light_robot[0],
            full_obs.player_has_heavy_robot[0],
            # navigation
            full_obs.rubble,
        ]
        full_skip_obs = [
            # where ore is
            full_obs.has_ore,
            # cargo status
            np.add(full_obs.player_light_robot_ore[0], full_obs.player_heavy_robot_ore[0]),
            # where factories are
            full_obs.player_has_factory[0],
        ]
    elif job == "courier":
        ...
    elif job == "sabateur":
        ...
    elif job == "scout":
        ...
    elif job == "soldier":
        ...
    elif job == "builder":
        ...
    elif job == "factory":
        ...
    else:
        raise ValueError(f"Unknown unit job: {job}")
    return full_conv_obs, full_skip_obs
