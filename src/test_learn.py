import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3


def test(sb3_algo, model):
    model_dir = "models"
    
    register(
        id='Robotis-v0',                      # call it whatever you want
        entry_point='robotis_env.robotis_env:RobotisEnv', # module_name:class_name
    )
    env = gym.make("Robotis-v0", render_mode="human", width=1920, height=1080)
    model = f"{model_dir}/{sb3_algo}_{model}"
    print(f"Testing model: {model}")

    match sb3_algo:
        case "SAC":
            model = SAC.load(model, env=env)
        case "TD3":
            model = TD3.load(model, env=env)
        case "A2C":
            model = A2C.load(model, env=env)
        case "PPO":
            model = PPO.load(model, env=env, device="cpu")
        case "DDPG":
            model = DDPG.load(model, env=env, device="cuda")
        case _:
            print("Algorithm not found")
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break