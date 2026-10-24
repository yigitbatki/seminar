# Stage 1 (Cal-ConRFT): offline pretraining on the demos. Exits when done.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.5}
cd "$(dirname "$0")/../.." && python train_conrft_octo.py \
    --exp_name=sim_pick_cube \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/conrft} \
    --q_weight=0.1 --bc_weight=1.0 \
    --demo_path=${DEMO_PATH:?set DEMO_PATH to the .pkl from run_record_demos.sh} \
    --pretrain_steps=${PRETRAIN_STEPS:-20000} \
    --debug=${DEBUG:-True} \
    --learner "$@"
