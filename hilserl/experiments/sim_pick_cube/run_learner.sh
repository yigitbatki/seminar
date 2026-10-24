# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.5}
cd "$(dirname "$0")/../.." && python train_rlpd.py \
    --exp_name=sim_pick_cube \
    --seed=${HILSERL_SIM_SEED:-42} \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/run1} \
    --demo_path=${DEMO_PATH:?set DEMO_PATH to the .pkl from run_record_demos.sh} \
    --debug=${DEBUG:-True} \
    --learner "$@"
