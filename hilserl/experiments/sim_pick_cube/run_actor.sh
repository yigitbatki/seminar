export HILSERL_SIM_MODE=${HILSERL_SIM_MODE:-online}
# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.2}
cd "$(dirname "$0")/../.." && python train_rlpd.py \
    --exp_name=sim_pick_cube \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/run1} \
    --debug=${DEBUG:-True} \
    --actor "$@"
