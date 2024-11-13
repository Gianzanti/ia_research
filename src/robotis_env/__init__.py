from gymnasium.envs.registration import register

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='Robotis-v0',                      # call it whatever you want
    entry_point='robotis_env:RobotisEnv', # module_name:class_name
)