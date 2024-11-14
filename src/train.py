import os
import time

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3


def train(sb3_algo):
    # Create directories to hold models and logs
    idx = int(time.time())
    model_dir = "models"
    log_dir = f"logs/{idx}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    register(
        id='Robotis-v0',                      # call it whatever you want
        entry_point='robotis_env.robotis_env:RobotisEnv', # module_name:class_name
    )
    env = gym.make("Robotis-v0")

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
            model = PPO(
                "MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir
            )

        case "DDPG":
            model = DDPG(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case _:
            print("Algorithm not found")
            return

    TIMESTEPS = 25000
    iters = 0
    # idx = time.time()
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True
        )
        model.save(f"{model_dir}/{sb3_algo}_{idx}_{TIMESTEPS*iters}")
        if iters == 1000:
            break