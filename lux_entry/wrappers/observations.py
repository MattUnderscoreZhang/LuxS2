import gym
from gym import spaces
import numpy as np
from typing import Dict


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        n_players = 2
        x, y = self.env.state.env_cfg.map_size, self.env.state.env_cfg.map_size
        self.env_cfg = self.env.state.env_cfg

        self.observation_space = spaces.Dict({
            "has_ice": spaces.MultiBinary((1, 1, x, y)),
            "has_ore": spaces.MultiBinary((1, 1, x, y)),
            "has_factory": spaces.MultiBinary((1, n_players, x, y)),
            "has_robot": spaces.MultiBinary((1, n_players, x, y)),
            "has_light_robot": spaces.MultiBinary((1, n_players, x, y)),
            "has_heavy_robot": spaces.MultiBinary((1, n_players, x, y)),
            "has_lichen": spaces.MultiBinary((1, n_players, x, y)),

            # -1 indicates no robot
            # normalize all values between 0-1, by dividing max possible value
            "normalized_rubble": spaces.Box(0., 1., shape=(1, 1, x, y)),
            "normalized_lichen": spaces.Box(0., 1., shape=(1, n_players, x, y)),
            "normalized_light_robot_power": spaces.Box(-1., 1., shape=(1, n_players, x, y)),
            "normalized_light_robot_ice": spaces.Box(-1., 1., shape=(1, n_players, x, y)),
            "normalized_light_robot_ore": spaces.Box(-1., 1., shape=(1, n_players, x, y)),
            "normalized_heavy_robot_power": spaces.Box(-1., 1., shape=(1, n_players, x, y)),
            "normalized_heavy_robot_ice": spaces.Box(-1., 1., shape=(1, n_players, x, y)),
            "normalized_heavy_robot_ore": spaces.Box(-1., 1., shape=(1, n_players, x, y)),

            # -1 indicates no factory
            # normalize these values by a reasonable amount, but allow the possibility of going over
            "normalized_factory_ice": spaces.Box(-1., 10., shape=(1, n_players, x, y)),
            "normalized_factory_ore": spaces.Box(-1., 10., shape=(1, n_players, x, y)),
            "normalized_factory_water": spaces.Box(-1., 10., shape=(1, n_players, x, y)),
            "normalized_factory_metal": spaces.Box(-1., 10., shape=(1, n_players, x, y)),
            "normalized_factory_power": spaces.Box(-1., 10., shape=(1, n_players, x, y)),

            # -1 indicates no factory
            # normalize these values by a reasonable amount, but allow the possibility of going over
            "total_normalized_robots": spaces.Box(0., 10., shape=(1, n_players)),
            "total_normalized_light_robots": spaces.Box(0., 10., shape=(1, n_players)),
            "total_normalized_heavy_robots": spaces.Box(0., 10., shape=(1, n_players)),
            "total_normalized_factories": spaces.Box(0., 10., shape=(1, n_players)),
            "total_normalized_factory_ice": spaces.Box(-1., 10., shape=(1, n_players)),
            "total_normalized_factory_ore": spaces.Box(-1., 10., shape=(1, n_players)),
            "total_normalized_factory_water": spaces.Box(-1., 10., shape=(1, n_players)),
            "total_normalized_factory_metal": spaces.Box(-1., 10., shape=(1, n_players)),
            "total_normalized_factory_power": spaces.Box(-1., 10., shape=(1, n_players)),
            "total_normalized_lichen": spaces.Box(0., 10., shape=(1, n_players)),

            # * environment values for is_day, phase_time_left, game_time_left, total_friendly_lichen, total_enemy_lichen
            # "is_night": spaces.MultiDiscrete(np.zeros((1, 1))),
            # "night": spaces.MultiDiscrete(np.zeros((1, 1)) + 2),
            # "day_night_cycle": spaces.MultiDiscrete(np.zeros((1, 1)) + DN_CYCLE_LEN),
            # "phase": spaces.MultiDiscrete(np.zeros((1, 1)) + GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN),
            # "turn": spaces.Box(0., 1., shape=(1, 1)),
            # "board_size": spaces.MultiDiscrete(np.zeros((1, 1)) + len(MAP_SIZES)),

            # TODO: robot actions
        })

    def observation(self, observation: Dict) -> Dict[str, np.ndarray]:
        max_factories = self.env_cfg.MAX_FACTORIES
        max_rubble = self.env_cfg.MAX_RUBBLE
        max_lichen = self.env_cfg.MAX_LICHEN_PER_TILE
        light_cargo_space = self.env_cfg.ROBOTS["LIGHT"].CARGO_SPACE
        light_battery_cap = self.env_cfg.ROBOTS["LIGHT"].BATTERY_CAPACITY
        heavy_cargo_space = self.env_cfg.ROBOTS["HEAVY"].CARGO_SPACE
        heavy_battery_cap = self.env_cfg.ROBOTS["HEAVY"].BATTERY_CAPACITY
        reasonable_factory_cargo_space = 2 * heavy_cargo_space
        ice_water_ratio = self.env_cfg.ICE_WATER_RATIO
        ore_metal_ratio = self.env_cfg.ORE_METAL_RATIO
        reasonable_factory_power = self.env_cfg.FACTORY_CHARGE * 30
        reasonable_max_robots = 50

        observation = observation["player_0"]  # for some reason, observation["player_0"] == observation["player_1"]
        obs = dict()
        for key, value in self.observation_space.items():
            if value.shape[1] > 1:
                obs[key] = (
                    np.zeros(value.shape[1:])
                    if type(value) == spaces.MultiBinary
                    else np.zeros(value.shape[1:]) + value.low[0]
                )
        obs["has_ice"] = observation["board"]["ice"]
        obs["has_ore"] = observation["board"]["ore"]
        obs["normalized_rubble"] = observation["board"]["rubble"] / max_rubble
        lichen_strains = [[], []]
        for p, player in enumerate(["player_0", "player_1"]):
            for factory in observation["factories"][player].values():
                lichen_strains[p].append(factory["strain_id"])
                obs["has_factory"][p][factory["pos"]] = 1
                obs["normalized_factory_ice"][p][factory["pos"]] = factory["cargo"]["ice"] / reasonable_factory_cargo_space
                obs["normalized_factory_ore"][p][factory["pos"]] = factory["cargo"]["ore"] / reasonable_factory_cargo_space
                obs["normalized_factory_water"][p][factory["pos"]] = factory["cargo"]["water"] / (reasonable_factory_cargo_space / ice_water_ratio)
                obs["normalized_factory_metal"][p][factory["pos"]] = factory["cargo"]["metal"] / (reasonable_factory_cargo_space / ore_metal_ratio)
                obs["normalized_factory_power"][p][factory["pos"]] = factory["power"] / reasonable_factory_power
                obs["total_normalized_factory_ice"][p] += factory["cargo"]["ice"] / (reasonable_factory_cargo_space * max_factories)
                obs["total_normalized_factory_ore"][p] += factory["cargo"]["ore"] / (reasonable_factory_cargo_space * max_factories)
                obs["total_normalized_factory_water"][p] += factory["cargo"]["water"] / (reasonable_factory_cargo_space * max_factories / ice_water_ratio)
                obs["total_normalized_factory_metal"][p] += factory["cargo"]["metal"] / (reasonable_factory_cargo_space * max_factories / ore_metal_ratio)
                obs["total_normalized_factory_power"][p] += factory["power"] / (reasonable_factory_power * max_factories)
            for robot in observation["units"][player].values():
                obs["has_robot"][p][robot["pos"]] = 1
                obs["has_light_robot"][p][robot["pos"]] = (robot["unit_type"] == "LIGHT")
                obs["has_heavy_robot"][p][robot["pos"]] = (robot["unit_type"] == "HEAVY")
                obs["normalized_light_robot_power"][p][robot["pos"]] = robot["power"] / light_battery_cap
                obs["normalized_light_robot_ice"][p][robot["pos"]] = robot["cargo"]["ice"] / light_cargo_space
                obs["normalized_light_robot_ore"][p][robot["pos"]] = robot["cargo"]["ore"] / light_cargo_space
                obs["normalized_heavy_robot_power"][p][robot["pos"]] = robot["power"] / heavy_battery_cap
                obs["normalized_heavy_robot_ice"][p][robot["pos"]] = robot["cargo"]["ice"] / heavy_cargo_space
                obs["normalized_heavy_robot_ore"][p][robot["pos"]] = robot["cargo"]["ore"] / heavy_cargo_space
                obs["total_normalized_robots"][p] += 1 / reasonable_max_robots
                obs["total_normalized_light_robots"][p] += 1/ reasonable_max_robots * (robot["unit_type"] == "LIGHT")
                obs["total_normalized_heavy_robots"][p] += 1/ reasonable_max_robots * (robot["unit_type"] == "HEAVY")
            obs["has_lichen"][p] = observation["board"]["lichen_strains"] == lichen_strains[p]
            obs["normalized_lichen"][p] = observation["board"]["lichen"] / max_lichen * (observation["board"]["lichen_strains"] == lichen_strains[p])
            obs["total_normalized_lichen"][p] = np.sum(obs["normalized_lichen"][p])
        obs["teams"] = observation["teams"]

        return obs
