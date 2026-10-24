# Record demos with the scripted expert (stands in for the SpaceMouse).
export HILSERL_SIM_MODE=demo
export HILSERL_SIM_HZ=${HILSERL_SIM_HZ:-0}  # no need to pace the scripted expert
# Colab sets MPLBACKEND to its inline backend, which the venv does not have.
export MPLBACKEND=Agg
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../.." && python record_demos.py --exp_name=sim_pick_cube --successes_needed=${N_DEMOS:-20} "$@"
