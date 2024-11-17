import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import TensorboardCallback


def train(sb3_algo):

    # Create directories to hold models and logs
    idx = int(time.time())
    model_dir = "models"
    log_dir = f"logs/{idx}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # register(
    #     id='Robotis-v0',                      # call it whatever you want
    #     entry_point='robotis_env.robotis_env:RobotisEnv', # module_name:class_name
    # )
    env = gym.make("Robotis-v0")
    TIMESTEPS = 25000

    match sb3_algo:
        case "SAC":
            model = SAC(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "TD3":
            model = TD3(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "A2C":
            model = A2C(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "PPO":
            env = make_vec_env("Robotis-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
            model = PPO(
                "MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir
            )

        case "DDPG":
            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,  device="cuda", tensorboard_log=log_dir)
            TIMESTEPS = 10000

        case _:
            print("Algorithm not found")
            return

    iters = 0
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback()
        )
        model.save(f"{model_dir}/{sb3_algo}_{idx}_{TIMESTEPS*iters}")
        if iters == 1000:
            break