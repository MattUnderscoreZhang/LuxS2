import argparse
import importlib
import os.path as osp
from pathlib import Path
import yaml

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder


class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose: int = 0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        assert self.logger is not None
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def train(args: argparse.Namespace, env_id: str, model: BaseAlgorithm):
    eval_env = SubprocVecEnv(
        [args.make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=args.eval_freq,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )
    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))


def evaluate(args: argparse.Namespace, env_id: str, model: BaseAlgorithm):
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [args.make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)


def main(args: argparse.Namespace):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"
    env = SubprocVecEnv(
        [
            args.make_env(env_id, i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    env.reset()
    if args.eval:
        evaluate(args, env_id, args.model(env, args))
    else:
        train(args, env_id, args.model(env, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--behavior", type=str, required=True, help="Behavior to train."
    )
    parser.add_argument(
        "-t",
        "--training-conf",
        type=str,
        default="test_conf",
        help="Training configurations.",
    )
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="If set, will put model in evaluation mode.",
    )
    args = parser.parse_args()

    with open(
        Path(__file__).parent / "train_configs" / (args.training_conf + ".yaml")
    ) as f:
        training_args = argparse.Namespace(**yaml.safe_load(f))
    training_args.model = importlib.import_module(
        f"lux_entry.behaviors.{args.behavior}.env"
    ).model
    training_args.make_env = importlib.import_module(
        f"lux_entry.behaviors.{args.behavior}.env"
    ).make_env
    training_args.log_path = (
        Path(__file__).parent / "behaviors" / args.behavior / "logs"
    )
    training_args.eval = args.eval
    training_args.model_path = (
        Path(__file__).parent
        / "behaviors"
        / args.behavior
        / "logs"
        / "models"
        / "best_model.zip"
    )
    main(training_args)
