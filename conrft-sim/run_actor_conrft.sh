# Stage 2 (HIL-ConRFT): online actor. The scripted expert takes over in a fraction of episodes.
export HILSERL_SIM_MODE=${HILSERL_SIM_MODE:-online}
# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.2}
PY="python"; [ "${CONRFT_TINY_OCTO:-0}" = 1 ] && PY="python tiny_octo_smoke.py"
# `yes ""` answers the "checkpoint exists, press Enter" prompt.
cd "$(dirname "$0")/../.." && yes "" | $PY train_conrft_octo.py \
    --exp_name=sim_pick_cube \
    --seed=${HILSERL_SIM_SEED:-42} \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/conrft} \
    --actor "$@"
