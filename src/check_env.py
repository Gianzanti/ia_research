def show_env():
    import gymnasium as gym

    print(f"Gym version: {gym.__version__}")

    import stable_baselines3

    print(f"Stable Baselines version: {stable_baselines3.__version__}")

    import mujoco

    print(f"Mujoco version: {mujoco.__version__}")

    env = gym.make("Humanoid-v5", render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()

    print("Environment check successful!")
