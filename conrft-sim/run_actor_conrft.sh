# Stage 2 (HIL-ConRFT): online actor. The scripted expert takes over in a fraction of episodes.
export CONRFT_SIM_MODE=online
export CONRFT_SIM_INTERVENE_PROB=${CONRFT_SIM_INTERVENE_PROB:-0.2}
# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.2}
# `yes ""` answers the "checkpoint exists, press Enter" prompt.
cd "$(dirname "$0")/../.." && yes "" | python train_conrft_octo.py \
    --exp_name=sim_pick_cube \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/conrft} \
    --actor "$@"
