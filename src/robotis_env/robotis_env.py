import os
from typing import Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.5,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -5.0,
}

def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

# you can completely modify this class for your MuJoCo environment by following the directions
class RobotisEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # set default episode_len for truncate episodes
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        frame_skip = 5

        self._forward_reward_weight: float = 1.25
        self._ctrl_cost_weight: float = 0.1
        self._contact_cost_weight: float = 5e-7
        self._contact_cost_range: Tuple[float, float] = (-np.inf, 10.0)
        self._healthy_reward: float = 5.0
        self._terminate_when_unhealthy: bool = True
        self._healthy_z_range: Tuple[float, float] = (0.20, 0.30)
        self._reset_noise_scale: float = 1e-2


        MujocoEnv.__init__(
            self,
            # os.path.abspath("src/robotis_op3/scene.xml"),
            os.path.join(os.path.dirname(__file__), "robotis_mjcf", "scene.xml"),
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size + self.data.sensordata.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )


    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.site('torso').xpos[2] < max_z
        return is_healthy

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward        


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        # contact_forces = self.data.cfrc_ext
        # contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        # min_cost, max_cost = self._contact_cost_range
        # contact_cost = np.clip(contact_cost, min_cost, max_cost)
        # return contact_cost
        return 0

    # determine the reward depending on observation or other properties of the simulation
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) # and self._terminate_when_unhealthy
        # terminated = False
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        # print(f"Data Joint {self.data.joint('head_pan').qpos[:3]}")
        # print(f"Data Joint {self.data.site('torso').xpos}")
        # print(f"Sensors {self.data.sensordata}")

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    def _get_rew(self, x_velocity: float, action):
        # print(f"Action: {action}")
        # print(f"X Velocity: {x_velocity}")
        # print(f"Forward Reward: {self._forward_reward_weight}")
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        self.step_number = 0

        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        noise_low = -0.01
        noise_high = 0.01

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        return self._get_obs()

    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        # obs = np.concatenate((np.array(self.data.joint("ball").qpos[:3]),
        #                       np.array(self.data.joint("ball").qvel[:3]),
        #                       np.array(self.data.joint("rotate_x").qpos),
        #                       np.array(self.data.joint("rotate_x").qvel),
        #                       np.array(self.data.joint("rotate_y").qpos),
        #                       np.array(self.data.joint("rotate_y").qvel)), axis=0)

        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        imu = self.data.sensordata.flatten()

        return np.concatenate(
            (
                position,
                velocity,
                imu,
                # com_inertia,
                # com_velocity,
                # actuator_forces,
                # external_contact_forces,
            )
        )
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }        