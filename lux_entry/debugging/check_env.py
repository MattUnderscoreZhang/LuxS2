from stable_baselines3.common.vec_env import SubprocVecEnv

from luxai_s2.map_generator.generator import argparse

from lux_entry.lux.utils import add_batch_dimension
from lux_entry.training.env import make_env
from lux_entry.training.net import get_model


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
    )
    model = get_model(env, args)

    for _ in range(100):
        obs = env.reset()
        step_n = 0
        total_reward = 0
        for _ in range(10):
            breakpoint()
            obs = add_batch_dimension(obs)
            action = 1
            obs, reward, done, _ = env.step(action)
            total_reward += reward

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
