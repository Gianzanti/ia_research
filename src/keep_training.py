import os

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from callback import TensorboardCallback


def keep_training(sb3_algo, model):
    idx = model
    model_dir = "models"
    log_dir = f"logs/{idx}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    model = f"{model_dir}/{sb3_algo}_{model}"
    print(f"Keep training model: {model}")

    register(
        id='Robotis-v0',                      # call it whatever you want
        entry_point='robotis_env.robotis_env:RobotisEnv', # module_name:class_name
    )
    env = gym.make("Robotis-v0")

    match sb3_algo:
        case "SAC":
            model = SAC.load(model, env=env, device="cuda", force_reset=True)
        case "TD3":
            model = TD3.load(model, env=env, device="cuda", force_reset=True)
        case "A2C":
            model = A2C.load(model, env=env, device="cuda", force_reset=True)
        case "PPO":
            model = PPO.load(model, env=env, device="cpu", force_reset=True)
        case "DDPG":
            model = DDPG.load(model, env=env, device="cuda", force_reset=True)
        case _:
            print("Algorithm not found")
            return


    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback()
        )
        model.save(f"{model_dir}/{sb3_algo}_{idx}_{TIMESTEPS*iters}")
        if iters == 1000:
            break