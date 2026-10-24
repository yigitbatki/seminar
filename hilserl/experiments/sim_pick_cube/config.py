import os

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
    """HIL-SERL (RLPD + interventions) on the simulated Franka cube-lifting task.

    The scripted expert replaces the SpaceMouse. Its behaviour is set with
    environment variables, so the upstream HIL-SERL scripts run unchanged:
      HILSERL_SIM_MODE=demo     expert controls every step (record_demos.py)
      HILSERL_SIM_MODE=online   expert takes over in some episodes (default)
      HILSERL_SIM_INTERVENE_PROB=0.2   probability of a takeover per episode
      HILSERL_SIM_EPISODE_LOG=path.csv  one row per actor episode
      HILSERL_SIM_TAKEOVER_STEPS=0      0: the expert keeps control to the end of the
                                        episode; N > 0: it hands back after N steps
      HILSERL_SIM_EXPERT_NOISE=0.1      std of the noise on the expert's xyz actions
      HILSERL_SIM_LIFT_HEIGHT=0.1       lift (m) that counts as success
      HILSERL_SIM_MAX_EPISODE_STEPS=100
      HILSERL_SIM_SEED=42               seeds the cube positions, the expert and (as --seed)
                                        train_rlpd.py's networks
      HILSERL_SIM_CUBE_RANGE=default    cube start area: default | wide (1.5x the area)
      HILSERL_SIM_DISCOUNT=0.97, HILSERL_SIM_CTA_RATIO=2 (critic updates per actor update),
      HILSERL_SIM_ENCODER=resnet-pretrained | resnet
      HILSERL_SIM_HZ=10         wall-clock cap on env steps/s, like the real Franka's 10 Hz
                                control loop; keeps about one learner update per env step
                                on fast GPUs (0 = no cap)
    HILSERL_SIM_BUFFER, HILSERL_SIM_BATCH_SIZE and HILSERL_SIM_CKPT_PERIOD override
    the replay buffer capacity, batch size and checkpoint period.
    """

    image_keys = ["front", "wrist"]
    classifier_keys = []
    proprio_keys = ["gripper_pose", "tcp_pose", "tcp_vel"]
    max_traj_length = int(os.environ.get("HILSERL_SIM_MAX_EPISODE_STEPS", "100"))
    # Each buffer slot holds two 128x128 images (~0.1 MB); the upstream default of
    # 200k slots would need ~20 GB per buffer, more than this machine has.
    replay_buffer_capacity = int(os.environ.get("HILSERL_SIM_BUFFER", "20000"))
    batch_size = int(os.environ.get("HILSERL_SIM_BATCH_SIZE", "256"))
    checkpoint_period = int(os.environ.get("HILSERL_SIM_CKPT_PERIOD", "2000"))
    buffer_period = 1000
    cta_ratio = int(os.environ.get("HILSERL_SIM_CTA_RATIO", "2"))
    discount = float(os.environ.get("HILSERL_SIM_DISCOUNT", "0.97"))
    random_steps = 0
    training_starts = 100
    steps_per_update = 50
    # "resnet-pretrained" = frozen ImageNet ResNet-10, "resnet" = ResNet-10 trained from scratch
    encoder_type = os.environ.get("HILSERL_SIM_ENCODER", "resnet-pretrained")
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # The seed also drives the cube positions and the expert's takeover/noise RNG.
        env_seed = int(os.environ.get("HILSERL_SIM_SEED", "42"))
        env = SimPickCubeEnv(
            max_episode_length=self.max_traj_length,
            lift_height=float(os.environ.get("HILSERL_SIM_LIFT_HEIGHT", "0.1")),
            control_hz=float(os.environ.get("HILSERL_SIM_HZ", "10")),
            cube_range=os.environ.get("HILSERL_SIM_CUBE_RANGE", "default"),
            seed=env_seed,
        )
        if not fake_env:
            env = SimExpertIntervention(
                env,
                mode=os.environ.get("HILSERL_SIM_MODE", "online"),
                episode_prob=float(os.environ.get("HILSERL_SIM_INTERVENE_PROB", "0.2")),
                expert_noise=float(os.environ.get("HILSERL_SIM_EXPERT_NOISE", "0.1")),
                takeover_steps=int(os.environ.get("HILSERL_SIM_TAKEOVER_STEPS", "0")),
                seed=env_seed,
            )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # The simulator gives the sparse success reward directly, so no reward classifier is needed.
        env = SimGripperPenaltyWrapper(env, penalty=-0.02)
        log_path = os.environ.get("HILSERL_SIM_EPISODE_LOG")
        if log_path and not fake_env:
            env = EpisodeLogWrapper(env, log_path)
        return env

    def process_demos(self, demo):
        return demo
