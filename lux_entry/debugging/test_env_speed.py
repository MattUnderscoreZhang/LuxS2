import argparse
from pathlib import Path
from time import time
import torch
from typing import Callable

from stable_baselines3.common.vec_env import SubprocVecEnv

from lux_entry.training.env import make_env
from lux_entry.training.model import (
    JobActionNet, JobNet, MapFeaturesExtractor, get_model, N_MAP_FEATURES
)
from lux_entry.training.observations import get_full_obs_space


def perform_timing(func: Callable, n_trials: int = 100, func_name: str = "") -> float:
    start = time()
    for _ in range(n_trials):
        func()
    end = time()
    run_time = (end - start) / n_trials * 1000
    if func_name != "":
        print(f"\n{func_name} runs in {run_time} ms")
    return run_time


def test_map_features_extractor():
    env = make_env(0)()
    full_obs_space = get_full_obs_space(env.state.env_cfg)
    batch_map_features = {
        key: torch.randn(6, value[1], 48, 48)
        for key, value in full_obs_space.spaces.items()
    }
    breakpoint()
    map_features_extractor = MapFeaturesExtractor(full_obs_space)
    func = lambda: map_features_extractor._forward(batch_map_features)
    run_time = perform_timing(func, func_name="MapFeaturesExtractor forward function")
    assert run_time < 10


def test_job_net():
    batch_map_features = torch.randn(6, N_MAP_FEATURES, 48, 48)
    job_net = JobNet()
    func = lambda: job_net(batch_map_features)
    run_time = perform_timing(func, func_name="JobNet forward function")
    assert run_time < 10


def test_job_action_net():
    batch_map_features = torch.randn(6, N_MAP_FEATURES, 48, 48)
    job_action_net = JobActionNet()
    func = lambda: job_action_net(batch_map_features)
    run_time = perform_timing(func, func_name="JobActionNet forward function")
    assert run_time < 10


def check_env() -> None:
    env = make_env(0)()
    env = SubprocVecEnv(
        [make_env(i, max_episode_steps=1000) for i in range(4)]
    )
    args = argparse.Namespace(
        seed=12,
        n_envs=1,
        max_episode_steps=200,
        total_timesteps=1,
        rollout_steps=3_000,
        eval_freq=24_000,
        batch_size=1_000,
        learning_rate=0.0003,
        target_kl=0.05,
        gamma=0.99,
        log_path=Path(__file__).parent / "logs",
    )
    model = get_model(env, args)

    for i in range(10):
        print("Episode", i)
        obs = env.reset()
        step_n = 0
        for _ in range(10):
            action = model.predict(obs)[0]
            obs, reward, done, _ = env.step(action)

            # # give title to entire figure
            # plt.figure()
            # plt.suptitle(f"Step {step_n}, Reward: {total_reward}, Action: {action}")
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

            step_n += 1


if __name__ == "__main__":
    check_env()
