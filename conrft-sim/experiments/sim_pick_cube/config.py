import os

import numpy as np

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.sim_pick_cube.sim_env import (
    SimPickCubeEnv,
    SimExpertIntervention,
    SimGripperPenaltyWrapper,
    EpisodeLogWrapper,
)


class TrainConfig(DefaultTrainingConfig):
    """ConRFT on the simulated Franka cube-lifting task.

    sim_env.py is the same file as HIL-SERL's (hilserl/experiments/sim_pick_cube), so both
    methods see the same task, cube positions, scripted expert and interventions. Task settings
    use the same HILSERL_SIM_* environment variables:
      HILSERL_SIM_MODE=demo|online, HILSERL_SIM_INTERVENE_PROB=0.2, HILSERL_SIM_TAKEOVER_STEPS=0,
      HILSERL_SIM_EXPERT_NOISE=0.1, HILSERL_SIM_SEED=42, HILSERL_SIM_CUBE_RANGE=default|wide,
      HILSERL_SIM_CUBE_POSITIONS=f.npy, HILSERL_SIM_LIFT_HEIGHT=0.1,
      HILSERL_SIM_MAX_EPISODE_STEPS=100, HILSERL_SIM_HZ=10, HILSERL_SIM_EPISODE_LOG=path.csv
    ConRFT's own settings use CONRFT_SIM_*: BUFFER (replay capacity), BATCH_SIZE, CKPT_PERIOD,
    and CONRFT_OCTO_PATH.
    Differences from HIL-SERL, as in ConRFT's code: a 256 px front camera ("side_policy_256") and
    a 128 px wrist camera ("wrist_1"), 2 stacked frames, reward +10 on success and -0.05 per step.
    """

    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = []
    proprio_keys = ["gripper_pose", "tcp_pose", "tcp_vel"]
    max_traj_length = int(os.environ.get("HILSERL_SIM_MAX_EPISODE_STEPS", "100"))
    checkpoint_period = int(os.environ.get("CONRFT_SIM_CKPT_PERIOD", "2000"))
    batch_size = int(os.environ.get("CONRFT_SIM_BATCH_SIZE", "256"))
    # Each buffer slot holds a 256x256 and a 128x128 image (~0.25 MB). The upstream
    # default of 200k slots needs ~45 GB per buffer, so keep it small.
    replay_buffer_capacity = int(os.environ.get("CONRFT_SIM_BUFFER", "10000"))
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    training_starts = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    success_reward = 10.0
    reward_neg = -0.05
    task_desc = "Pick up the red cube and lift it"
    # Octo-small from Hugging Face. Point this to a local copy if you have one.
    octo_path = os.environ.get("CONRFT_OCTO_PATH", "hf://rail-berkeley/octo-small")

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        seed = int(os.environ.get("HILSERL_SIM_SEED", "42"))
        positions = os.environ.get("HILSERL_SIM_CUBE_POSITIONS")
        env = SimPickCubeEnv(
            max_episode_length=self.max_traj_length,
            lift_height=float(os.environ.get("HILSERL_SIM_LIFT_HEIGHT", "0.1")),
            success_reward=self.success_reward,
            reward_neg=self.reward_neg,
            image_size=256,
            wrist_size=128,
            image_keys=tuple(self.image_keys),
            control_hz=float(os.environ.get("HILSERL_SIM_HZ", "10")),
            cube_range=os.environ.get("HILSERL_SIM_CUBE_RANGE", "default"),
            cube_positions=np.load(positions) if positions else None,
            seed=seed,
        )
        if not fake_env:
            env = SimExpertIntervention(
                env,
                mode=os.environ.get("HILSERL_SIM_MODE", "online"),
                episode_prob=float(os.environ.get("HILSERL_SIM_INTERVENE_PROB", "0.2")),
                expert_noise=float(os.environ.get("HILSERL_SIM_EXPERT_NOISE", "0.1")),
                takeover_steps=int(os.environ.get("HILSERL_SIM_TAKEOVER_STEPS", "0")),
                seed=seed,
            )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        # The simulator gives the sparse success reward directly, so no reward classifier is needed.
        env = SimGripperPenaltyWrapper(env, penalty=-0.02)
        log_path = os.environ.get("HILSERL_SIM_EPISODE_LOG")
        if log_path and not fake_env:
            env = EpisodeLogWrapper(env, log_path)
        return env
