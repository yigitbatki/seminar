#!/usr/bin/env bash
# Short ConRFT test with the real Octo-small: 2 demos, PRETRAIN_STEPS pretraining steps, then
# ONLINE_STEPS online env steps. Prints peak GPU memory, pretraining steps/s and actor steps/s,
# so you can check that ConRFT fits the GPU before starting the real runs. Nothing is kept.
# usage: [WORKDIR=~/conrft-sim-work] [TEST_DIR=/content/octo_test] bash octo_test.sh
set -u
WORKDIR=${WORKDIR:-$HOME/conrft-sim-work}
TEST_DIR=${TEST_DIR:-/content/octo_test}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-1000}
ONLINE_STEPS=${ONLINE_STEPS:-300}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf "$TEST_DIR"; mkdir -p "$TEST_DIR"
export WORKDIR MPLBACKEND=Agg
unset HILSERL_SIM_EPISODE_LOG HILSERL_SIM_CUBE_POSITIONS

( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$TEST_DIR/gpu_mem.log" &
MEM=$!
peak() { sort -n "$TEST_DIR/gpu_mem.log" | tail -1; }

echo "== recording 2 demos"
( source "$WORKDIR/.venv/bin/activate" && unset PYTHONPATH && cd "$WORKDIR/conrft/examples" && rm -f demo_data/sim_pick_cube_2_demos_*.pkl \
  && N_DEMOS=2 bash experiments/sim_pick_cube/run_record_demos.sh 2>&1 | grep -aE "saved|Error|Traceback" )
DEMO=$(ls "$WORKDIR"/conrft/examples/demo_data/sim_pick_cube_2_demos_*.pkl 2>/dev/null | tail -1)
[ -n "$DEMO" ] || { echo "demo recording failed"; kill $MEM; exit 1; }
echo "peak GPU memory so far: $(peak) MiB"

echo "== pretraining $PRETRAIN_STEPS steps (Cal-ConRFT)"
T0=$(date +%s)
STAGE=pretrain DEMO_PATH="$DEMO" RUN_DIR="$TEST_DIR/pretrain" PRETRAIN_STEPS=$PRETRAIN_STEPS CONRFT_SIM_CKPT_PERIOD=$PRETRAIN_STEPS \
  bash "$HERE/run_conrft.sh" | grep -E "pretrain step|finished|exited|Traceback|Error"
T1=$(date +%s)
echo "pretraining wall time $((T1 - T0)) s (includes loading Octo and compiling); peak GPU memory so far: $(peak) MiB"

echo "== online $ONLINE_STEPS env steps (HIL-ConRFT, learner + actor)"
STAGE=online DEMO_PATH="$DEMO" RUN_DIR="$TEST_DIR/online" INIT_CKPT="$TEST_DIR/pretrain/ckpt/checkpoint_$PRETRAIN_STEPS" \
  TARGET_ENV_STEPS=$ONLINE_STEPS CONRFT_SIM_CKPT_PERIOD=100000 bash "$HERE/run_conrft.sh" \
  | grep -E "env steps|reached|exited|died|Traceback|Error"
kill $MEM
echo "== summary"
echo "peak GPU memory (whole test): $(peak) MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MiB"
echo "(the online lines above show env steps/s and learner updates/s once both are warm)"
