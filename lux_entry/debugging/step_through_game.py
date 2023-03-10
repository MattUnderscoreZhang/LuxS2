import argparse
import importlib
import io
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import yaml
import zipfile

from lux_entry.components.types import PolicyNet
from lux_entry.lux.utils import add_batch_dimension


def load_net(model_class: PolicyNet, model_path: str) -> PolicyNet:
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
    net.load_weights(sb3_state_dict)
    return net


def step_through_game(args: argparse.Namespace) -> None:
    env = args.env.make_env(0)()
    net: args.net.Net = load_net(args.net.Net, args.net.WEIGHTS_PATH)

    for _ in range(100):
        obs = env.reset()
        done = False
        i = 0
        total_reward = 0
        while not done:
            obs = add_batch_dimension(obs)
            action = net.evaluate(obs, deterministic=False)
            action = action.cpu().numpy()[0]
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # # give title to entire figure
            # plt.figure()
            # plt.suptitle(f"Step {i}, Reward: {total_reward}, Action: {action}")
            # plt.axis("off")
            # plt.subplot(2, 2, 1)
            # plt.imshow(obs["skip_obs"][0])
            # plt.clim(-1, 1)
            # plt.subplot(2, 2, 2)
            # plt.imshow(obs["skip_obs"][1])
            # plt.clim(-1, 1)
            # plt.subplot(2, 2, 3)
            # plt.imshow(obs["skip_obs"][2])
            # plt.clim(-1, 1)
            # plt.subplot(2, 2, 4)
            # plt.imshow(obs["skip_obs"][3])
            # plt.clim(-1, 1)
            # plt.show()

            i += 1


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

    step_through_game(args)
