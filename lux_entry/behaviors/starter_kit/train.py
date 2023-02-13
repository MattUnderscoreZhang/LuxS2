import argparse
import gym
from gym.wrappers.time_limit import TimeLimit
import os.path as osp
from typing import Callable

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO

from lux_entry.behaviors.starter_kit import wrappers
from lux_entry.heuristics import bidding, factory_placement


def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps: int = 100) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
        env = wrappers.MainGameOnlyWrapper(
            env,
            bid_policy=bidding.zero_bid,
            factory_placement_policy=factory_placement.place_near_random_ice,
            controller=wrappers.ControllerWrapper(env.env_cfg),
        )
        env = wrappers.ObservationWrapper(env)  # changes observation to include a few simple features
        env = wrappers.EnvWrapper(env)  # convert to single agent, add our reward
        env = TimeLimit(env, max_episode_steps=max_episode_steps)  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init


class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose: int = 0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True


def train(args: argparse.Namespace, env_id: str, model: BaseAlgorithm):
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(4)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
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
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
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
            make_env(env_id, i, max_episode_steps=args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )
    env.reset()
    rollout_steps = 4000
    policy_kwargs = dict(net_arch=(128, 128))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=800,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.99,
        tensorboard_log=osp.join(args.log_path),
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training.")
    parser.add_argument("-n", "--n-envs", type=int, default=8, help="Number of parallel envs to run. Separate from rollout size.")
    parser.add_argument("-m", "--max-episode-steps", type=int, default=200, help="Max steps per episode before truncating them.")
    parser.add_argument("-t", "--total-timesteps", type=int, default=3_000_000, help="Total timesteps for training.")
    parser.add_argument("-e", "--eval", action="store_true", help="If set, will only evaluate a given policy. Otherwise enters training mode.")
    parser.add_argument("-p", "--model-path", type=str, help="Path to SB3 model weights to use for evaluation.")
    parser.add_argument("-l", "--log-path", type=str, default="logs", help="Logging path.")
    main(parser.parse_args())
