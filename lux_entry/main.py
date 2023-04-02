from argparse import Namespace
import io
import json
import numpy as np
import torch
from typing import Any, Union
import zipfile

from luxai_s2.state.state import ObservationStateDict
from lux_entry.heuristics.factory_placement import FactoryPlacementActionType

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.utils import (
    add_batch_dimension,
    my_turn_to_place_factory,
    process_action,
    process_obs,
)
from lux_entry.training import env, net
from lux_entry.train import WEIGHTS_PATH


class Agent:
    def __init__(self, player: Player, env_cfg: EnvConfig) -> None:
        self.player: Player = player
        self.env_cfg: EnvConfig = env_cfg

        self.model: net.UnitsNet = self._load_model(net.UnitsNet, WEIGHTS_PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)
        self.controller: env.EnvController = env.EnvController(self.env_cfg)

    def _load_model(self, model_class: type[net.UnitsNet], model_path: str) -> net.UnitsNet:
        if model_path[-4:] == ".zip":
            with zipfile.ZipFile(model_path) as archive:
                file_path = "policy.pth"
                with archive.open(file_path, mode="r") as param_file:
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    file_content.seek(0)
                    sb3_state_dict = torch.load(file_content, map_location="cpu")
        else:
            sb3_state_dict = torch.load(model_path, map_location="cpu")

        model = model_class()
        model.load_weights(sb3_state_dict)
        return model

    def bid_policy(
        self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60
    ):
        return env.bid_policy(player=self.player, obs=obs)

    def factory_placement_policy(
        self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> FactoryPlacementActionType:
        return (
            env.factory_placement_policy(player=self.player, obs=obs)
            if my_turn_to_place_factory(
                obs["teams"][self.player]["place_first"],
                step,
            )
            else FactoryPlacementActionType(
                metal=0, water=0, spawn=np.array([]),
            )  # empty action since it's not our turn
        )

    def act(
        self, step: int, env_obs: ObservationStateDict, remainingOverageTime: int = 60
    ) -> dict[str, int]:
        obs = ObservationWrapper.get_obs(env_obs, self.env_cfg, self.player)

        with torch.no_grad():
            action_mask = add_batch_dimension(
                self.controller.action_masks(player=self.player, obs=env_obs)
            ).bool()
            observation = add_batch_dimension(obs)
            actions = (
                self.model(observation, deterministic=False, action_masks=action_mask)
                .cpu()
                .numpy()
            )
        return self.controller.action_to_lux_action(self.player, env_obs, actions[0])


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict: dict[Player, Agent] = dict()
agent_prev_obs: dict[Player, Union[ObservationStateDict, None]] = dict()


Json = Any


def agent_fn(observation: Namespace, configurations: dict) -> Json:
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg: EnvConfig = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = None
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    if step == 0:
        actions = agent.bid_policy(step, obs, remainingOverageTime)
    elif obs["real_env_steps"] < 0:
        actions = agent.factory_placement_policy(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    configurations = None
    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)
        observation = Namespace(
            **dict(
                step=obs["step"],
                obs=json.dumps(obs["obs"]),
                remainingOverageTime=obs["remainingOverageTime"],
                player=obs["player"],
                info=obs["info"],
            )
        )
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))
