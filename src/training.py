import os
import time

import gymnasium as gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import TensorboardCallback


def transfer(sb3_algo, model):
    idx = int(time.time())
    model_dir = "models"
    log_dir = f"logs/{idx}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model = f"{model_dir}/{sb3_algo}_{model}"
    print(f"Transfer learning from model: {model}")

    env = gym.make("Robotis-v0")
    TIMESTEPS = 25000

    match sb3_algo:
        case "SAC":
            model = SAC.load(model, env=env, device="cuda", force_reset=True)
        case "TD3":
            model = TD3.load(model, env=env, device="cuda", force_reset=True)
        case "A2C":
            model = A2C.load(model, env=env, device="cuda", force_reset=True)
        case "PPO":
            env = make_vec_env("Robotis-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
            model = PPO.load(model, env=env, device="cpu", force_reset=True)
        case "DDPG":
            model = DDPG.load(model, env=env, device="cuda", force_reset=True)
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