from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from termcolor import colored
from typing import Dict, List, Literal, Union

from lux_entry.lux.config import EnvConfig, UnitConfig


# Player = Union[Literal["player_0"], Literal["player_1"]]
Player = str


class Direction:
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


class FactionTypes(Enum):
    Null = FactionInfo(color="gray", faction_id=0)
    AlphaStrike = FactionInfo(color="yellow", faction_id=1)
    MotherMars = FactionInfo(color="green", faction_id=2)
    TheBuilders = FactionInfo(color="blue", faction_id=3)
    FirstMars = FactionInfo(color="red", faction_id=4)


class Team:
    def __init__(
        self,
        team_id: int,
        agent: str,
        faction: FactionTypes,
        water=0,
        metal=0,
        factories_to_place=0,
        factory_strains=[],
        place_first=False,
        bid=0,
    ) -> None:
        self.faction = faction
        self.team_id = team_id
        # the key used to differentiate ownership of things in state
        self.agent = agent

        self.water = water
        self.metal = metal
        self.factories_to_place = factories_to_place
        self.factory_strains = factory_strains
        # whether this team gets to place factories down first or not. The bid winner has this set to True.
        # If tied, player_0's team has this True
        self.place_first = place_first

    def state_dict(self):
        return dict(
            team_id=self.team_id,
            faction=self.faction.name,
            # note for optimization, water,metal, factories_to_place doesn't change after the early game.
            water=self.water,
            metal=self.metal,
            factories_to_place=self.factories_to_place,
            factory_strains=self.factory_strains,
            place_first=self.place_first,
        )

    def __str__(self) -> str:
        out = f"[Player {self.team_id}]"
        return colored(out, self.faction.value.color)


@dataclass
class Cargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: Cargo
    env_cfg: EnvConfig
    unit_cfg: UnitConfig
    action_queue: List

    @property
    def agent_id(self) -> Player:
        if self.team_id == 0:
            return "player_0"
        return "player_1"

    def action_queue_cost(self, game_state: 'GameState'):
        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    def move_cost(self, game_state: 'GameState', direction):
        board = game_state.board
        target_pos = self.pos + move_deltas[direction]
        if (
            target_pos[0] < 0
            or target_pos[1] < 0
            or target_pos[1] >= len(board.rubble)
            or target_pos[0] >= len(board.rubble[0])
        ):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if (
            factory_there not in game_state.teams[self.agent_id].factory_strains
            and factory_there != -1
        ):
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(
            self.unit_cfg.MOVE_COST
            + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
        )

    def move(self, direction, repeat: int = 0, n: int = 1):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, repeat, n])

    def transfer(
        self, transfer_direction, transfer_resource: int, transfer_amount: int, repeat: int = 0, n: int = 1
    ):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0
        return np.array(
            [1, transfer_direction, transfer_resource, transfer_amount, repeat, n]
        )

    def pickup(self, pickup_resource: int, pickup_amount: int, repeat: int = 0, n: int = 1):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, n])

    def dig_cost(self, game_state: 'GameState'):
        return self.unit_cfg.DIG_COST

    def dig(self, repeat: int = 0, n: int = 1):
        return np.array([3, 0, 0, 0, repeat, n])

    def self_destruct_cost(self, game_state: 'GameState'):
        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat: int = 0, n: int = 1):
        return np.array([4, 0, 0, 0, repeat, n])

    def recharge(self, x, repeat: int = 0, n: int = 1):
        return np.array([5, 0, 0, x, repeat, n])

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out


@dataclass
class Factory:
    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: Cargo
    pos: np.ndarray
    # lichen tiles connected to this factory
    # lichen_tiles: np.ndarray
    env_cfg: EnvConfig

    def build_heavy_metal_cost(self, game_state: 'GameState'):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST

    def build_heavy_power_cost(self, game_state: 'GameState'):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST

    def can_build_heavy(self, game_state: 'GameState'):
        return self.power >= self.build_heavy_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)

    def build_heavy(self):
        return 1

    def build_light_metal_cost(self, game_state: 'GameState'):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST

    def build_light_power_cost(self, game_state: 'GameState'):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.POWER_COST

    def can_build_light(self, game_state: 'GameState'):
        return self.power >= self.build_light_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        return 0

    def water_cost(self, game_state: 'GameState'):
        """
        Water required to perform water action
        """
        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR)

    def can_water(self, game_state: 'GameState'):
        return self.cargo.water >= self.water_cost(game_state)

    def water(self):
        return 2

    @property
    def pos_slice(self):
        return slice(self.pos[0] - 1, self.pos[0] + 2), slice(
            self.pos[1] - 1, self.pos[1] + 2
        )


@dataclass
class Board:
    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    lichen: np.ndarray
    lichen_strains: np.ndarray
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    valid_spawns_mask: np.ndarray


@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from lux_entry.luxai_s2/state/state.py
    """

    env_steps: int
    env_cfg: EnvConfig
    board: Board
    units: Dict[str, Dict[str, Unit]] = field(default_factory=dict)
    factories: Dict[str, Dict[str, Factory]] = field(default_factory=dict)
    teams: Dict[str, Team] = field(default_factory=dict)

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for extra factory placement and + 1 for bidding step
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps

    # various utility functions
    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH
