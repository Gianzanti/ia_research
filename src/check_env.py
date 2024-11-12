def show_env():
    import gymnasium as gym

    print(f"Gym version: {gym.__version__}")

    import stable_baselines3

    print(f"Stable Baselines version: {stable_baselines3.__version__}")

    import mujoco
    import os

    print(f"Mujoco version: {mujoco.__version__}")

    xml_file = os.path.join(os.path.dirname(__file__), "robotis_op3", "scene.xml")
    env = gym.make("Humanoid-v5", xml_file=xml_file, render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    # while True:
    #     env.render()

    env.close()

    print("Environment check successful!")
