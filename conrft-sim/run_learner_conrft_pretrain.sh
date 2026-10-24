# Stage 1: offline pretraining on the demos. Exits when done.
# Cal-ConRFT: Q_WEIGHT=0.1 BC_WEIGHT=1.0 (default); SFT baseline: Q_WEIGHT=0 BC_WEIGHT=1.0.
# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.5}
PY="python"; [ "${CONRFT_TINY_OCTO:-0}" = 1 ] && PY="python tiny_octo_smoke.py"
cd "$(dirname "$0")/../.." && $PY train_conrft_octo.py \
    --exp_name=sim_pick_cube \
    --seed=${HILSERL_SIM_SEED:-42} \
    --checkpoint_path=${CKPT:-$PWD/experiments/sim_pick_cube/conrft} \
    --q_weight=${Q_WEIGHT:-0.1} --bc_weight=${BC_WEIGHT:-1.0} \
    --demo_path=${DEMO_PATH:?set DEMO_PATH to the .pkl from run_record_demos.sh} \
    --pretrain_steps=${PRETRAIN_STEPS:-20000} \
    --debug=${DEBUG:-True} \
    --learner "$@"
