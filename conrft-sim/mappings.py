from experiments.sim_pick_cube.config import TrainConfig as SimPickCubeTrainConfig

CONFIG_MAPPING = {
    "sim_pick_cube": SimPickCubeTrainConfig,
}

# The real-robot task needs serl_robot_infra (franka_env, ROS, SpaceMouse).
# Keep it available when that is installed, without breaking the simulator setup.
try:
    from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig

    CONFIG_MAPPING["task1_pick_banana"] = PickBananaTrainConfig
except ImportError:
    pass
