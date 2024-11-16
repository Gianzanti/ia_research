import gymnasium as gym


def show_env():
    print(f"Gym version: {gym.__version__}")

    import stable_baselines3

    print(f"Stable Baselines version: {stable_baselines3.__version__}")

    import mujoco

    print(f"Mujoco version: {mujoco.__version__}")

    env = gym.make("Robotis-v0", render_mode="human")

    # it will check your custom environment and output additional warnings if needed
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    env.close()

    print("Environment check successful!")

def check_model():
    env = gym.make('Robotis-v0', render_mode="human", width=1920, height=1080)
    
    observation, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Info: {info}")
    print(f"Action sample: {env.action_space.sample()}")

    episode_over = False
    counter = 0
    action_one = 0.00
    increment = 0.01
    step = increment
    action = [0] * env.action_space.shape[0]
    while not episode_over:
        if counter % 100 == 0:
            action[0] = action_one
            action[1] = action_one
            action[2] = action_one
            action[3] = action_one
            action[4] = action_one
            action[5] = action_one

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = counter > 300000
            
            if (action_one > 3.14):
                step = -increment
            if (action_one < -3.14):
                step = increment

            action_one += step
            

            # print(f"Info: {info}")
        counter += 1
        # episode_over = False

    env.close()

    print("Environment check successful!")
