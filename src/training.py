import os
import time

import gymnasium as gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from callback import TensorboardCallback


def make_dirs(sb3_algo, timestep=None):
    if timestep:
        idx = timestep
    else:
        idx = int(time.time())
    model_dir = f"models/{sb3_algo}_{idx}"
    log_dir = f"logs/{idx}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return model_dir, log_dir


def transfer(sb3_algo, model):
    model_dir, log_dir = make_dirs(sb3_algo)
    
    from_model_dir = model.split("_")[0]
    from_model_timesteps = model.split("_")[1]
    model_file = f"models/{sb3_algo}_{from_model_dir}/{sb3_algo}_{from_model_dir}_{from_model_timesteps}"
    print(f"Transfer learning from model: {model_file} to {model_dir}")

    env = gym.make("Robotis-v0")
    TIMESTEPS = 25000

    match sb3_algo:
        case "SAC":
            model = SAC.load(model_file, env=env, device="cuda", force_reset=True)
        case "TD3":
            model = TD3.load(model_file, env=env, device="cuda", force_reset=True)
        case "A2C":
            model = A2C.load(model_file, env=env, device="cuda", force_reset=True)
        case "PPO":
            env = make_vec_env("Robotis-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
            model = PPO.load(model_file, env=env, device="cpu", force_reset=True)
        case "DDPG":
            model = DDPG.load(model_file, env=env, device="cuda", force_reset=True)
        case _:
            print("Algorithm not found")
            return

    iters = 0
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback()
        )
        model.save(f"{model_dir}/{TIMESTEPS*iters}")
        if iters == 1000:
            break


def keep_training(sb3_algo, model):
    from_model_dir = model.split("_")[0]
    from_model_timesteps = model.split("_")[1]
    model_dir, log_dir = make_dirs(sb3_algo, from_model_dir)
    print(f"Keep training model: {model_dir}")
    print(f"Keep training log: {log_dir}")

    model_file = f"models/{sb3_algo}_{from_model_dir}/{from_model_timesteps}"
    print(f"Keep training from model: {model_file}")

    print(f"Model save: {model_dir}/TIMESTEPS")

    env = gym.make("Robotis-v0")
    TIMESTEPS = 25000

    match sb3_algo:
        case "SAC":
            model = SAC.load(model_file, env=env, device="cuda", force_reset=True)
        case "TD3":
            model = TD3.load(model_file, env=env, device="cuda", force_reset=True)
        case "A2C":
            model = A2C.load(model_file, env=env, device="cuda", force_reset=True)
        case "PPO":
            env = make_vec_env("Robotis-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
            model = PPO.load(model_file, env=env, device="cpu", force_reset=True)
        case "DDPG":
            model = DDPG.load(model_file, env=env, device="cuda", force_reset=True)
        case _:
            print("Algorithm not found")
            return

    iters = int(from_model_timesteps) // TIMESTEPS
    while True:
        iters += 1
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback()
        )
        model.save(f"{model_dir}/{TIMESTEPS*iters}")
        if iters == 10000:
            break


def train(sb3_algo):
    model_dir, log_dir = make_dirs(sb3_algo)

    env = gym.make("Robotis-v0")
    TIMESTEPS = 25000

    learn_args = {
        "total_timesteps": TIMESTEPS, 
        "reset_num_timesteps": False, 
        "progress_bar":True, 
        "callback": TensorboardCallback()
    }

    match sb3_algo:
        case "SAC":
            model = SAC(
                "MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir
            )
        case "TD3":
            import numpy as np
            from stable_baselines3.common.noise import NormalActionNoise
            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,  device="cuda", tensorboard_log=log_dir)



        case "A2C":
            from stable_baselines3.common.sb2_compat.rmsprop_tf_like import (
                RMSpropTFLike,
            )
            env = make_vec_env("Robotis-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
            model = A2C("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))

        case "PPO":
            env = make_vec_env("Robotis-v0", n_envs=2, vec_env_cls=SubprocVecEnv)
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                device="cpu", 
                tensorboard_log=log_dir, 
                # batch_size= 16,
                # clip_range= 0.4,
                # ent_coef= 3.241790106161843e-07,
                # gae_lambda= 0.98,
                # gamma= 0.999,
                # learning_rate= 0.33193338832282054,
                # max_grad_norm= 2,
                # n_epochs= 10,
                # n_steps= 2048,
                # vf_coef= 0.2745982707731268,
                # # policy_kwargs = {
                # #     'ortho_init': False,
                # #     'activation_fn': 'tanh',
                # #     'net_arch': 'medium'
                # # }            
            )

        case "DDPG":
            import numpy as np
            from stable_baselines3.common.noise import NormalActionNoise
            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,  device="cuda", tensorboard_log=log_dir)


        case "ARS":
            from sb3_contrib import ARS
            # from sb3_contrib.common.vec_env import AsyncEval
            # n_envs = 2
            model = ARS("LinearPolicy", env, n_delta=2, n_top=1, verbose=1, device="cuda", tensorboard_log=log_dir)
            # # Create env for asynchronous evaluation (run in different processes)
            # async_eval = AsyncEval([lambda: make_vec_env(env) for _ in range(n_envs)], model.policy)
            # learn_args["async_eval"] = async_eval
            learn_args["log_interval"] = 4

            # model.learn(log_interval=4, async_eval=async_eval)            

        case "TRPO":
            from sb3_contrib import TRPO
            env = make_vec_env("Robotis-v0", n_envs=4, vec_env_cls=SubprocVecEnv)
            model = TRPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir)
            learn_args["log_interval"] = 4

        case _:
            print("Algorithm not found")
            return

    iters = 0
    while True:
        iters += 1
        # model.learn(
        #     total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, callback=TensorboardCallback()
        # )
        model.learn(**learn_args)
        model.save(f"{model_dir}/{TIMESTEPS*iters}")
        if iters == 1000:
            break


def test(sb3_algo, model):
    from_model_dir = model.split("_")[0]
    from_model_timesteps = model.split("_")[1]
    model_dir, log_dir = make_dirs(sb3_algo, from_model_dir)    

    model = f"{model_dir}/{from_model_timesteps}"

    env = gym.make("Robotis-v0", render_mode="human", width=1920, height=1080)
    print(f"Testing model: {model}")

    match sb3_algo:
        case "SAC":
            model = SAC.load(model, env=env)
        case "TD3":
            model = TD3.load(model, env=env)
        case "A2C":
            model = A2C.load(model, env=env, device="cpu")
        case "PPO":
            model = PPO.load(model, env=env, device="cpu")
        case "DDPG":
            model = DDPG.load(model, env=env, device="cuda")
        case "TRPO":
            from sb3_contrib import TRPO
            model = TRPO.load(model, env=env, device="cpu")

        case _:
            print("Algorithm not found")
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 200
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break