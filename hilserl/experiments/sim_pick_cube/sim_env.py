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

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import franka_sim.envs.panda_pick_gym_env as panda_pick
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.mujoco_gym_env import GymRenderingSpec


# Cube start area (x, y in metres) and the matching reach limits of the arm's target point.
# "default" is franka_sim's own setting; "wide" covers 1.5x the area: farther out to the sides and
# front while the cube stays fully inside the front camera image. Closer to the robot base than
# x = 0.27 (near y = 0) the arm cannot reach down to the cube, so the near edge stays at 0.25.
CUBE_RANGES = {
    "default": dict(sampling=[[0.25, -0.25], [0.55, 0.25]], cartesian=[[0.2, -0.3, 0], [0.6, 0.3, 0.5]]),
    "wide": dict(sampling=[[0.25, -0.32], [0.60, 0.32]], cartesian=[[0.15, -0.40, 0], [0.70, 0.40, 0.5]]),
}


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
        control_hz=0.0,
        cube_range="default",
        seed=None,
        image_keys=("front", "wrist"),
        wrist_size=None,
        cube_positions=None,
    ):
        """image_size is the front camera's (and rendering) resolution; wrist_size, if different,
        resizes the wrist camera. image_keys names the two cameras in the observation (ConRFT
        uses "side_policy_256" at 256 px and "wrist_1" at 128 px). cube_positions, an (N, 2)
        array of cube x, y, replaces the random cube positions (used in order, cycling)."""
        # franka_sim reads these module-level bounds on every reset/step.
        panda_pick._SAMPLING_BOUNDS = np.asarray(CUBE_RANGES[cube_range]["sampling"])
        panda_pick._CARTESIAN_BOUNDS = np.asarray(CUBE_RANGES[cube_range]["cartesian"])
        # franka_sim samples the cube with numpy's global RNG; keep a private, seeded copy of it.
        self._cube_rng_state = np.random.RandomState(0 if seed is None else seed).get_state()
        self._env = PandaPickCubeGymEnv(
            action_scale=np.asarray(action_scale),
            render_spec=GymRenderingSpec(height=image_size, width=image_size),
            render_mode="rgb_array",
            image_obs=True,
            time_limit=1e6,  # episode length is handled here instead
            seed=0 if seed is None else seed,
        )
        self.front_key, self.wrist_key = image_keys
        self.wrist_size = wrist_size or image_size
        self.cube_positions = None if cube_positions is None else np.asarray(cube_positions, dtype=np.float64)
        self._cube_index = 0
        self.max_episode_length = max_episode_length
        self.lift_height = lift_height
        self.success_reward = success_reward
        self.reward_neg = reward_neg
        self._t = 0
        # Wall-clock cap on the step rate, like the real robot's control loop (0 = as fast as possible).
        self.control_hz = control_hz
        self._next_step_time = 0.0

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
                        self.front_key: spaces.Box(
                            0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
                        ),
                        self.wrist_key: spaces.Box(
                            0, 255, shape=(self.wrist_size, self.wrist_size, 3), dtype=np.uint8
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
        wrist = obs["images"]["wrist"]
        if wrist.shape[0] != self.wrist_size:
            wrist = cv2.resize(wrist, (self.wrist_size, self.wrist_size), interpolation=cv2.INTER_AREA)
        return {
            "state": {
                "gripper_pose": np.atleast_1d(obs["state"]["panda/gripper_pos"]).astype(np.float32),
                "tcp_pose": obs["state"]["panda/tcp_pos"].astype(np.float32),
                "tcp_vel": obs["state"]["panda/tcp_vel"].astype(np.float32),
            },
            "images": {
                self.front_key: np.ascontiguousarray(obs["images"]["front"], dtype=np.uint8),
                self.wrist_key: np.ascontiguousarray(wrist, dtype=np.uint8),
            },
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._cube_rng_state = np.random.RandomState(seed).get_state()
        outer = np.random.get_state()
        np.random.set_state(self._cube_rng_state)
        obs, _ = self._env.reset()
        self._cube_rng_state = np.random.get_state()
        np.random.set_state(outer)
        if self.cube_positions is not None:
            xy = self.cube_positions[self._cube_index % len(self.cube_positions)]
            self._cube_index += 1
            sim = self._env
            sim._data.jnt("block").qpos[:3] = (*xy, sim._block_z)
            mujoco.mj_forward(sim._model, sim._data)
            sim._z_init = sim._data.sensor("block_pos").data[2]
            obs = sim._compute_observation()
        self._t = 0
        return self._convert_obs(obs), {"succeed": False}

    def step(self, action):
        if self.control_hz > 0:
            now = time.monotonic()
            if now < self._next_step_time:
                time.sleep(self._next_step_time - now)
            self._next_step_time = max(now, self._next_step_time) + 1.0 / self.control_hz
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
                   random step, like a human correcting a stuck robot. It keeps
                   control to the end of the episode (takeover_steps=0) or hands
                   back after `takeover_steps` steps. Set episode_prob=0 for no help.
    """

    def __init__(self, env, mode="online", episode_prob=0.2, expert_noise=0.1, takeover_steps=0, seed=0):
        super().__init__(env)
        self.mode = mode
        self.episode_prob = episode_prob
        self.takeover_steps = takeover_steps
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
        if intervened and self.mode != "demo" and self.takeover_steps > 0:
            intervened = self._t < self._takeover_step + self.takeover_steps
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
                csv.writer(f, lineterminator="\n").writerow(["episode", "time", "return", "succeed", "length", "intervened_steps"])
        with open(path) as f:
            self._episode = sum(1 for _ in f) - 1  # continue numbering after a resume

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
                csv.writer(f, lineterminator="\n").writerow([self._episode, f"{time.time():.1f}", self._return,
                                        int(info["succeed"]), self._length, self._intervened])
            self._episode += 1
        return obs, reward, terminated, truncated, info
