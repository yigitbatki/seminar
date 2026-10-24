"""Simulated Franka cube-lifting task for HIL-SERL.

Wraps SERL's MuJoCo `franka_sim` PandaPickCube environment so that it looks like
HIL-SERL's real-robot environment: gymnasium API, two 128x128 cameras ("front"
and "wrist", the size HIL-SERL's ResNet-10 encoder uses), a sparse 0/1 success
reward with the episode ending on success (what the reward classifier wrapper
does on the real robot), `info["succeed"]`, and `info["intervene_action"]` from a
scripted expert that stands in for the human with the SpaceMouse.
"""
import copy
import csv
import os
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.mujoco_gym_env import GymRenderingSpec


class SimPickCubeEnv(gym.Env):
    """Image-based cube lifting with a sparse reward.

    Action: [dx, dy, dz, drx, dry, drz, grasp] in [-1, 1], the same 7-D layout as
    HIL-SERL's real Franka with a learned gripper (the grasp critic reads the last entry). The simulated gripper
    keeps a fixed orientation, so the rotation entries are ignored. Position deltas
    are scaled by `action_scale[0]` metres; grasp > 0.5 closes the gripper, < -0.5 opens it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_length=100,
        lift_height=0.1,
        success_reward=1.0,
        reward_neg=0.0,
        action_scale=(0.05, 1.0),
        image_size=128,
        seed=None,
    ):
        self._env = PandaPickCubeGymEnv(
            action_scale=np.asarray(action_scale),
            render_spec=GymRenderingSpec(height=image_size, width=image_size),
            render_mode="rgb_array",
            image_obs=True,
            time_limit=1e6,  # episode length is handled here instead
            seed=0 if seed is None else seed,
        )
        self.max_episode_length = max_episode_length
        self.lift_height = lift_height
        self.success_reward = success_reward
        self.reward_neg = reward_neg
        self._t = 0

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "gripper_pose": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
                "images": spaces.Dict(
                    {
                        "front": spaces.Box(
                            0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
                        ),
                        "wrist": spaces.Box(
                            0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    # Privileged state for the scripted expert and success checks.
    @property
    def block_pos(self):
        return self._env._data.sensor("block_pos").data.copy()

    @property
    def tcp_pos(self):
        return self._env._data.sensor("2f85/pinch_pos").data.copy()

    @property
    def gripper_pos(self):
        return float(self._env._data.ctrl[self._env._gripper_ctrl_id] / 255)

    @property
    def z_init(self):
        return self._env._z_init

    def _convert_obs(self, obs):
        return {
            "state": {
                "gripper_pose": np.atleast_1d(obs["state"]["panda/gripper_pos"]).astype(np.float32),
                "tcp_pose": obs["state"]["panda/tcp_pos"].astype(np.float32),
                "tcp_vel": obs["state"]["panda/tcp_vel"].astype(np.float32),
            },
            "images": {
                "front": np.ascontiguousarray(obs["images"]["front"], dtype=np.uint8),
                "wrist": np.ascontiguousarray(obs["images"]["wrist"], dtype=np.uint8),
            },
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, _ = self._env.reset()
        self._t = 0
        return self._convert_obs(obs), {"succeed": False}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        grasp = action[-1]
        sim_action = np.array(
            [*np.clip(action[:3], -1.0, 1.0), 1.0 if grasp > 0.5 else (-1.0 if grasp < -0.5 else 0.0)],
            dtype=np.float32,
        )
        obs, _, _, _, _ = self._env.step(sim_action)
        self._t += 1

        succeed = bool(self.block_pos[2] - self.z_init >= self.lift_height)
        reward = self.success_reward if succeed else self.reward_neg
        # Like HIL-SERL's real FrankaEnv: the episode is "done" on success or on timeout,
        # and truncated stays False (record_demos.py only checks `done`).
        terminated = succeed or self._t >= self.max_episode_length
        return self._convert_obs(obs), reward, terminated, False, {"succeed": succeed}


class ScriptedPickExpert:
    """Privileged pick-and-lift controller used as the simulated human.

    It steers the mocap target (which the arm's operational-space controller
    tracks) rather than the measured gripper position, which avoids overshoot.
    """

    def __init__(self, env: SimPickCubeEnv, gain=1.0, noise=0.0, rng=None):
        self.env = env
        self.gain = gain
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.close_steps = 0

    def act(self):
        env = self.env
        block, tcp = env.block_pos, env.tcp_pos
        mocap = env._env._data.mocap_pos[0].copy()
        grasp = 0.0
        if self.phase == "approach":
            target = block + np.array([0.0, 0.0, 0.08])
            grasp = -1.0
            if np.linalg.norm((block - tcp)[:2]) < 0.008 and abs(target[2] - tcp[2]) < 0.02:
                self.phase = "descend"
        if self.phase == "descend":
            target = block.copy()
            grasp = -1.0
            if np.linalg.norm(block - tcp) < 0.01:
                self.phase = "close"
        if self.phase == "close":
            target = mocap
            grasp = 1.0
            self.close_steps += 1
            if self.close_steps >= 6:
                self.phase = "lift"
        if self.phase == "lift":
            target = np.array([mocap[0], mocap[1], env.z_init + env.lift_height + 0.1])
            grasp = 0.0
        scale = env._env._action_scale[0]
        delta = np.clip(self.gain * (target - mocap) / scale, -1.0, 1.0)
        if self.noise > 0:
            delta = np.clip(delta + self.rng.normal(0.0, self.noise, size=3), -1.0, 1.0)
        return np.concatenate([delta, np.zeros(3), [grasp]]).astype(np.float32)


class SimExpertIntervention(gym.Wrapper):
    """Stands in for HIL-SERL's SpacemouseIntervention.

    mode="demo":   the expert controls every step (for recording demos).
    mode="online": with probability `episode_prob`, the expert takes over from a
                   random step until the end of the episode, like a human
                   correcting a stuck robot. Set episode_prob=0 for no help.
    """

    def __init__(self, env, mode="online", episode_prob=0.2, expert_noise=0.1, seed=0):
        super().__init__(env)
        self.mode = mode
        self.episode_prob = episode_prob
        self.rng = np.random.default_rng(seed)
        self.expert = ScriptedPickExpert(env.unwrapped, noise=expert_noise, rng=self.rng)
        self._takeover_step = None
        self._t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.expert.reset()
        self._t = 0
        if self.mode == "demo":
            self._takeover_step = 0
        elif self.rng.random() < self.episode_prob:
            self._takeover_step = int(self.rng.integers(0, max(1, self.env.unwrapped.max_episode_length // 2)))
        else:
            self._takeover_step = None
        return obs, info

    def step(self, action):
        intervened = self._takeover_step is not None and self._t >= self._takeover_step
        if intervened:
            action = self.expert.act()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if intervened:
            info["intervene_action"] = copy.deepcopy(action)
        return obs, reward, terminated, truncated, info


class SimGripperPenaltyWrapper(gym.Wrapper):
    """Adds `info["grasp_penalty"]` when the gripper command toggles its state, like
    HIL-SERL's GripperPenaltyWrapper, read by the grasp critic."""

    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.env.unwrapped.gripper_pos
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        executed = info.get("intervene_action", action)
        grasp = float(np.asarray(executed)[-1])
        toggled = (grasp > 0.5 and self.last_gripper_pos < 0.5) or (
            grasp < -0.5 and self.last_gripper_pos > 0.5
        )
        info["grasp_penalty"] = self.penalty if toggled else 0.0
        self.last_gripper_pos = self.env.unwrapped.gripper_pos
        return obs, reward, terminated, truncated, info


class EpisodeLogWrapper(gym.Wrapper):
    """Appends one CSV row per episode (return, success, length, intervened steps)."""

    def __init__(self, env, path):
        super().__init__(env)
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "time", "return", "succeed", "length", "intervened_steps"])
        self._episode = 0

    def reset(self, **kwargs):
        self._return, self._length, self._intervened = 0.0, 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._length += 1
        self._intervened += "intervene_action" in info
        if terminated or truncated:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([self._episode, f"{time.time():.1f}", self._return,
                                        int(info["succeed"]), self._length, self._intervened])
            self._episode += 1
        return obs, reward, terminated, truncated, info
