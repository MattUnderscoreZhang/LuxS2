import argparse
from pathlib import Path
import yaml

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder

from lux_entry.training.env import make_env
from lux_entry.training.net import get_model


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


def train(args: argparse.Namespace, model: BaseAlgorithm):
    eval_env = SubprocVecEnv(
        [make_env(i, max_episode_steps=1000) for i in range(4)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.log_path / "models",
        log_path=args.log_path / "eval_logs",
        eval_freq=args.eval_freq,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )
    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(args.log_path / "models" / "latest_model")


def evaluate(args: argparse.Namespace, model: BaseAlgorithm):
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [make_env(i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        args.log_path / "eval_videos",
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
    env = SubprocVecEnv(  # TODO: pass things like reward functions into make_env for each lesson in curriculum
        [
            make_env(i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    env.reset()
    model = get_model(env, args)
    if args.eval:
        evaluate(args, model)
    else:
        if args.continue_training:
            model = model.load(args.model_path)
            model.set_env(env)
        train(args, model)  # TODO: instead of setting up model and training once, set up models for each lesson in the curriculum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
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
    parser.add_argument(
        "--new_training",
        action="store_true",
        help="Start training anew, ignoring the best existing weights",
    )
    args = parser.parse_args()

    with open(
        Path(__file__).parent / "train_configs" / (args.training_conf + ".yaml")
    ) as f:
        training_args = argparse.Namespace(**yaml.safe_load(f))
    training_args.log_path = Path(__file__).parent / "logs"
    training_args.model_path = Path(__file__).parent / "logs" / "models" / "best_model"
    training_args.eval = args.eval
    training_args.continue_training = not args.new_training
    main(training_args)
