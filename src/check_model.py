# import os

import gymnasium as gym

# from robotis_env import RobotisEnv


def check_model():
    # xml_file = os.path.join(os.path.dirname(__file__), "robotis_op3", "scene.xml")
    # env = gym.make("Humanoid-v5", xml_file=xml_file, render_mode="human", width=1920, height=1080)
    from gymnasium.envs.registration import register

    # Register this module as a gym environment. Once registered, the id is usable in gym.make().
    register(
        id='Robotis-v0',                      # call it whatever you want
        entry_point='robotis_env.robotis_env:RobotisEnv', # module_name:class_name
    )
    env = gym.make("Robotis-v0", render_mode="human", width=1920, height=1080)
    
    observation, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Info: {info}")

    print(f"Action sample: {env.action_space.sample()}")

    episode_over = False
    counter = 0
    action_one = 0.00
    step = 0.005
    while not episode_over:
        # action = env.action_space.sample()
        if counter % 100 == 0:
            action = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                # 0,
                # 0,
            ]
            observation, reward, terminated, truncated, info = env.step(action)
            # episode_over = terminated or truncated
            episode_over = counter > 300000
            if (action_one > 3.14):
                step = -0.005
            if (action_one < -3.14):
                step = 0.005
            action_one += step

            print(f"Info: {info}")




        counter += 1
        
        episode_over = False

    # while True:
    #     env.render()

    env.close()

    print("Environment check successful!")
