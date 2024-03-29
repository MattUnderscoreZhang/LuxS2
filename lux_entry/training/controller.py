from gym import spaces
import numpy as np
from typing import Dict

from luxai_s2.state.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player


# TODO: add all actions
class EnvController:
    def __init__(self, env_cfg: EnvConfig) -> None:
        """
        Set the action space, convert actions to Lux actions, and calculate action masks.

        This simple controller controls only a single robot.
        It will always try to spawn one heavy robot if there are none.

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        For more info, see the lux action space definition in luxai_s2/spaces/action.py
        """
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        map_size = env_cfg.map_size
        self.map_size = map_size
        # TODO: make this take a single-unit action space, along with a location tuple
        self.action_space = spaces.MultiDiscrete([self.total_act_dims] * map_size * map_size)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def actions_to_lux_actions(
        self, player: Player, obs: ObservationStateDict, actions: np.ndarray
    ) -> Dict[str, int]:
        # TODO: make this select the action for a single unit and collect a dictionary
        # get units and sort by x position, then y position
        # this makes the units order consistent with the actions passed from the model
        units = obs["units"][player]
        units = dict(
            sorted(
                units.items(),
                key=lambda x: (x[1]["pos"][0], x[1]["pos"][1]),
            )
        )
        actions = actions.reshape(self.map_size, self.map_size)

        # get action for each unit
        lux_actions = dict()
        for unit_id, unit in units.items():
            action_queue = []
            no_op = False
            pos = unit["pos"]
            action = actions[pos[0], pos[1]]
            if self._is_move_action(action):
                action_queue = [self._get_move_action(action)]
            elif self._is_transfer_action(action):
                action_queue = [self._get_transfer_action(action)]
            elif self._is_pickup_action(action):
                action_queue = [self._get_pickup_action(action)]
            elif self._is_dig_action(action):
                action_queue = [self._get_dig_action(action)]
            else:
                no_op = True

            # simple trick to help units conserve power is to avoid updating the action queue
            # if the unit was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_actions[unit_id] = action_queue

        # get action for each factory
        factories = obs["factories"][player]
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_actions[unit_id] = 1  # build a single heavy

        # print(
            # "FACTORIES: " + str(list(factories.keys())) + "\n" +
            # "UNITS: " + str(list(units.keys())) + "\n" +
            # "ACTIONS: " + str(lux_actions) + "\n"
        # )
        return lux_actions

    def action_masks(self, player: Player, obs: ObservationStateDict) -> np.ndarray:
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        factory_occupancy_map = (
            np.ones_like(obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in obs["factories"]:
            factories[player] = dict()
            for unit_id in obs["factories"][player]:
                f_data = obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = obs["units"][player]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in obs["teams"][player]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in obs["teams"][player]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                obs["board"]["ice"][pos[0], pos[1]]
                + obs["board"]["ore"][pos[0], pos[1]]
                + obs["board"]["rubble"][pos[0], pos[1]]
                + obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask
