# Record demos with the scripted expert (stands in for the SpaceMouse).
export CONRFT_SIM_MODE=demo
export MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$(dirname "$0")/../.." && python record_demos_octo.py --exp_name=sim_pick_cube --successes_needed=${N_DEMOS:-20} "$@"
