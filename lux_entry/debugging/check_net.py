import argparse
import importlib
import io
from pathlib import Path
import torch
import yaml
import zipfile

from stable_baselines3.common.vec_env import SubprocVecEnv


def check_net(args: argparse.Namespace) -> None:
    if str(args.model_path)[-4:] == ".zip":
        with zipfile.ZipFile(args.model_path) as archive:
            file_path = "policy.pth"
            with archive.open(file_path, mode="r") as param_file:
                file_content = io.BytesIO()
                file_content.write(param_file.read())
                file_content.seek(0)
                sb3_state_dict = torch.load(file_content, map_location="cpu")
    else:
        sb3_state_dict = torch.load(args.model_path, map_location="cpu")

    net_keys = sb3_state_dict.keys()
    for key in net_keys:
        print(key, sb3_state_dict[key].shape)

    net = args.net.Net()
    loaded_state_dict = {}
    for sb3_key, model_key in zip(net_keys, net.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key)


def check_model(args: argparse.Namespace) -> None:
    my_env = SubprocVecEnv(
        [
            args.env.make_env(i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    model = args.net.model(my_env, args)
    print(model.policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--behavior", type=str, required=True, help="Behavior to train."
    )
    behavior = parser.parse_args().behavior

    root = Path(__file__).parent.parent
    with open(root / "train_configs" / "debug_conf.yaml") as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    args.net = importlib.import_module(f"lux_entry.behaviors.{behavior}.net")
    args.env = importlib.import_module(f"lux_entry.behaviors.{behavior}.env")
    args.log_path = root / "behaviors" / behavior / "logs"
    args.model_path = args.log_path / "models" / "best_model.zip"

    check_net(args)
    check_model(args)
