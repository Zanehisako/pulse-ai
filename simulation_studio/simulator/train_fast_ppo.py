# train_fast_ppo.py

from fast_env import FastBloodEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env():
    return FastBloodEnv(n_envs=1)


def main():
    env = DummyVecEnv([make_env for _ in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # smaller
        n_steps=1024,  # more stable
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.10,  # more exploration
        verbose=1,
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_fast_model")


if __name__ == "__main__":
    main()
