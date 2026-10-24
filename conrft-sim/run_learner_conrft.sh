# Stage 2 (HIL-ConRFT): online learner. Start this, then run_actor_conrft.sh in a second process.
# CKPT must hold the stage-1 checkpoint checkpoint_<PRETRAIN_STEPS> (or later online checkpoints).
# HIL-ConRFT: Q_WEIGHT=1.0 BC_WEIGHT=0.1 (default); HG-DAgger: Q_WEIGHT=0 BC_WEIGHT=1 BC_DEMO_ONLY=1.
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
    --q_weight=${Q_WEIGHT:-1.0} --bc_weight=${BC_WEIGHT:-0.1} \
    --bc_demo_only=$([ "${BC_DEMO_ONLY:-0}" = 1 ] && echo True || echo False) \
    --demo_path=${DEMO_PATH:?set DEMO_PATH to the .pkl from run_record_demos.sh} \
    --pretrain_steps=${PRETRAIN_STEPS:-20000} \
    --debug=${DEBUG:-True} \
    --learner "$@"
