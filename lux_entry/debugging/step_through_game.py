import argparse
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from lux_entry.main import Agent


def step_through_game(args: argparse.Namespace) -> None:
    player = "player_0"
    env = args.env.make_env(0)()
    agent = Agent(player, env.state.env_cfg)

    obs = env.reset()
    done = False
    i = 0
    while not done:
        obs = {
            key: value.float().unsqueeze(0)  # adding batch dimension
            for key, value in obs.items()
        }
        action = agent.net.act(obs, deterministic=False)
        action = action.cpu().numpy()[0]
        obs, reward, done, info = env.step(action)

        # plt.figure()
        # plt.title(f"Step {i}")
        # plt.axis("off")
        # plt.subplot(2, 2, 1)
        # plt.imshow(obs["skip_obs"][0])
        # plt.subplot(2, 2, 2)
        # plt.imshow(obs["skip_obs"][1])
        # plt.subplot(2, 2, 3)
        # plt.imshow(obs["skip_obs"][2])
        # plt.subplot(2, 2, 4)
        # plt.imshow(obs["skip_obs"][3])
        # # plt.show()

        print(reward)
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
