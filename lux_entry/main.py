from argparse import Namespace
import io
import json
import sys
import torch
import zipfile

from luxai_s2.state.state import ObservationStateDict

from lux_entry.lux.config import EnvConfig
from lux_entry.lux.state import Player
from lux_entry.lux.utils import my_turn_to_place_factory, process_action, process_obs

# change this to import a different behavior
from lux_entry.behaviors.starter_kit import model, env


class Agent:
    def __init__(self, player: Player, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = load_net(model.Net, model.WEIGHTS_PATH)
        self.net.eval().to(device)

        self.controller = env.controller.Controller(self.env_cfg)

    def bid_policy(self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60):
        return env.bid_policy(player=self.player, obs=obs)

    def factory_placement_policy(self, step: int, obs: ObservationStateDict, remainingOverageTime: int = 60):
        return (
            env.factory_placement_policy(player=self.player, obs=obs)
            if my_turn_to_place_factory(
                obs["teams"][self.player]["place_first"],
                step,
            )
            else dict()
        )

    def act(self, step: int, env_obs: ObservationStateDict, remainingOverageTime: int = 60):
        two_player_env_obs = {
            "player_0": env_obs,
            "player_1": env_obs,
        }
        obs = env.observation_wrapper.ObservationWrapper.get_custom_obs(two_player_env_obs, env_cfg=self.env_cfg)

        with torch.no_grad():
            action_mask = (
                torch.from_numpy(self.controller.action_masks(self.player, two_player_env_obs))
                .unsqueeze(0)  # we unsqueeze/add an extra batch dimension =
                .bool()
            )
            obs_arr = torch.from_numpy(obs[self.player]).float()
            actions = (
                self.net.act(
                    obs_arr.unsqueeze(0), deterministic=False, action_masks=action_mask
                )
                .cpu()
                .numpy()
            )
        return self.controller.action_to_lux_action(self.player, two_player_env_obs, actions[0])


def load_net(model_class: type[model.Net], model_path: str) -> model.Net:
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

    net = model_class()
    loaded_state_dict = {}

    # this code here works assuming the first keys in the sb3 state dict are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), net.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)

    net.load_state_dict(loaded_state_dict)
    return net


### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
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
