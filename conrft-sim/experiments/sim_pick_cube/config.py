import os

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.sim_pick_cube.sim_env import (
    SimPickCubeEnv,
    SimExpertIntervention,
    SimGripperPenaltyWrapper,
)


class TrainConfig(DefaultTrainingConfig):
    """ConRFT on the simulated Franka cube-lifting task.

    The scripted expert replaces the SpaceMouse. Its behaviour is set with
    environment variables, so the upstream ConRFT scripts run unchanged:
      CONRFT_SIM_MODE=demo     expert controls every step (record_demos_octo.py)
      CONRFT_SIM_MODE=online   expert takes over in some episodes (default)
      CONRFT_SIM_INTERVENE_PROB=0.2   probability of a takeover per episode
    CONRFT_SIM_CKPT_PERIOD and CONRFT_SIM_BATCH_SIZE override the checkpoint
    period and batch size (useful for quick tests), and CONRFT_SIM_BUFFER the
    replay buffer capacity.
    """

    image_keys = ["side_policy_256", "wrist_1"]
    classifier_keys = []
    proprio_keys = ["gripper_pose", "tcp_pose", "tcp_vel"]
    max_traj_length = 100
    checkpoint_period = int(os.environ.get("CONRFT_SIM_CKPT_PERIOD", "2000"))
    batch_size = int(os.environ.get("CONRFT_SIM_BATCH_SIZE", "256"))
    # Each buffer slot holds a 256x256 and a 128x128 image (~0.25 MB). The upstream
    # default of 200k slots needs ~45 GB per buffer, so keep it small for Colab.
    replay_buffer_capacity = int(os.environ.get("CONRFT_SIM_BUFFER", "10000"))
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    training_starts = 100
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    reward_neg = -0.05
    task_desc = "Pick up the red cube and lift it"
    # Octo-small from Hugging Face. Point this to a local copy if you have one.
    octo_path = os.environ.get("CONRFT_OCTO_PATH", "hf://rail-berkeley/octo-small")

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        env = SimPickCubeEnv(
            max_episode_length=self.max_traj_length,
            reward_neg=self.reward_neg,
        )
        if not fake_env:
            env = SimExpertIntervention(
                env,
                mode=os.environ.get("CONRFT_SIM_MODE", "online"),
                episode_prob=float(os.environ.get("CONRFT_SIM_INTERVENE_PROB", "0.2")),
            )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        # The simulator gives the sparse success reward directly, so no reward classifier is needed.
        env = SimGripperPenaltyWrapper(env, penalty=-0.02)
        return env
