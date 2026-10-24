# HIL-SERL sim smoke test: learner + actor on one GPU, stopped at MAX_ENV_STEPS or MAX_MIN minutes.
# usage: DEMO_PATH=... [WORKDIR=~/hilserl-work] bash smoke_test.sh RUN_NAME
set -u
RUN=${1:-smoke1}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-2000}
MAX_MIN=${MAX_MIN:-20}
WORKDIR=${WORKDIR:-$HOME/hilserl-work}
LOGS=$WORKDIR/logs/$RUN
mkdir -p $LOGS
cd $WORKDIR/hil-serl/examples
source $WORKDIR/.venv/bin/activate
unset PYTHONPATH
export CKPT=$PWD/experiments/sim_pick_cube/$RUN
export HILSERL_SIM_CKPT_PERIOD=${HILSERL_SIM_CKPT_PERIOD:-500}
EPLOG=$LOGS/episodes.csv
[ -e "$CKPT" ] && { echo "checkpoint dir $CKPT exists, pick a new RUN"; exit 1; }

( while true; do echo "$(date +%s) $(free -m | awk '/Mem/{print $3, $7}') $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"; sleep 5; done ) > $LOGS/mem.log &
MEM=$!
XLA_PYTHON_CLIENT_MEM_FRACTION=${LEARNER_FRAC:-.72} bash experiments/sim_pick_cube/run_learner.sh > $LOGS/learner.log 2>&1 &
LP=$!
until grep -q "starting learner loop" $LOGS/learner.log || ! kill -0 $LP 2>/dev/null; do sleep 2; done
kill -0 $LP 2>/dev/null || { echo "learner died"; kill $MEM; exit 1; }
echo "learner up $(date +%T)"
T0=$(date +%s)
HILSERL_SIM_EPISODE_LOG=$EPLOG XLA_PYTHON_CLIENT_MEM_FRACTION=${ACTOR_FRAC:-.15} bash experiments/sim_pick_cube/run_actor.sh > $LOGS/actor.log 2>&1 &
AP=$!
while true; do
  sleep 10
  steps=$( [ -f $EPLOG ] && awk -F, 'NR>1{s+=$5} END{print s+0}' $EPLOG || echo 0)
  el=$(( $(date +%s) - T0 ))
  kill -0 $LP 2>/dev/null || { echo "learner exited"; break; }
  kill -0 $AP 2>/dev/null || { echo "actor exited"; break; }
  [ $steps -ge $MAX_ENV_STEPS ] && { echo "reached $steps env steps in ${el}s"; break; }
  [ $el -ge $(( MAX_MIN * 60 )) ] && { echo "time limit, $steps env steps (completed episodes) in ${el}s"; break; }
done
pkill -INT -P $AP 2>/dev/null; pkill -INT -P $LP 2>/dev/null; sleep 5
pkill -P $AP 2>/dev/null; pkill -P $LP 2>/dev/null; kill $AP $LP 2>/dev/null; sleep 2
kill $MEM
echo "done $(date +%T)"
