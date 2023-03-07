from argparse import Namespace
import io
import json
import sys
import torch
from typing import Any, Dict, Union
import zipfile

from luxai_s2.state.state import ObservationStateDict

from lux_entry.components.types import Controller
from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.utils import my_turn_to_place_factory, process_action, process_obs

# change this to import a different behavior
from lux_entry.behaviors.starter_kit import env, net


class Agent:
    def __init__(self, player: Player, env_cfg: EnvConfig) -> None:
        self.player: Player = player
        self.env_cfg: EnvConfig = env_cfg

        self.net: net.Net = self._load_net(net.Net, net.WEIGHTS_PATH)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.eval().to(device)
        self.controller: Controller = env.EnvController(self.env_cfg)

    def _load_net_second_method(self, model_class: type[net.Net], model_path: str) -> net.Net:
        # TODO: this doesn't work yet
        from stable_baselines3 import PPO
        net = model_class()
        ppo = PPO.load(
            model_path,
            policy_kwargs={
                "features_extractor_class": net.CustomFeatureExtractor,
            }
        )
        state_dict = ppo.policy.state_dict()
        net.load_state_dict(state_dict)
        net.eval()
        return net

    def _load_net(self, model_class: type[net.Net], model_path: str) -> net.Net:
        # TODO: try replacing function with evaluate() in train.py
        # load .pth or .zip
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

        net_keys = [
            key
            for key in sb3_state_dict.keys()
            if key.startswith("features_extractor.net.features_net")
        ]
        net_keys += [
            key
            for key in sb3_state_dict.keys()
            if key.startswith("action_net.")
        ]

        net = model_class()
        loaded_state_dict = {}
        for sb3_key, model_key in zip(net_keys, net.state_dict().keys()):
            loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
            print("loaded", sb3_key, "->", model_key, file=sys.stderr)

        net.load_state_dict(loaded_state_dict)
        return net

    def bid_policy(
        self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60
    ):
        return env.bid_policy(player=self.player, obs=obs)

    def factory_placement_policy(
        self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60
    ):
        return (
            env.factory_placement_policy(player=self.player, obs=obs)
            if my_turn_to_place_factory(
                obs["teams"][self.player]["place_first"],
                step,
            )
            else dict()  # empty action since it's not our turn
        )

    def act(
        self, step: int, env_obs: ObservationStateDict, remainingOverageTime: int = 60
    ):
        return env.act(
            step=step,
            env_obs=env_obs,
            remainingOverageTime=remainingOverageTime,
            player=self.player,
            env_cfg=self.env_cfg,
            controller=self.controller,
            net=self.net,
        )


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict: Dict[Player, Agent] = dict()
agent_prev_obs: Dict[Player, Union[ObservationStateDict, None]] = dict()


Json = Any


def agent_fn(observation: Namespace, configurations: Dict) -> Json:
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
