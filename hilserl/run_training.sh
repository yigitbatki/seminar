#!/usr/bin/env bash
# Resumable HIL-SERL training on sim_pick_cube: learner + actor on one GPU.
# Re-running it with the same RUN_DIR (or SYNC_DIR) continues from the latest checkpoint.
#
# usage: DEMO_PATH=... RUN_DIR=... [SYNC_DIR=...] [TARGET_ENV_STEPS=30000] bash run_training.sh
#   WORKDIR           where setup.sh put hil-serl and .venv (default ~/hilserl-work)
#   RUN_DIR           checkpoints, buffer dumps, episodes.csv and logs (fast local disk)
#   SYNC_DIR          optional copy of RUN_DIR (e.g. Google Drive), refreshed every SYNC_MIN minutes
#                     and after each new checkpoint; used to restore RUN_DIR when it is missing
#   KEEP_CKPT=2       checkpoints kept (each ~0.3 GB)
#   KEEP_BUFFERS=10   actor buffer dumps kept (1000 env steps, ~0.19 GB each; only for resuming)
#   LEARNER_FRAC=.72  ACTOR_FRAC=.15   GPU memory caps (6 GB card; use .6 / .25 on a 16 GB T4)
#   HILSERL_SIM_CKPT_PERIOD=1000       learner steps between checkpoints
set -u
WORKDIR=${WORKDIR:-$HOME/hilserl-work}
: "${DEMO_PATH:?set DEMO_PATH to the demo .pkl}"
: "${RUN_DIR:?set RUN_DIR}"
SYNC_DIR=${SYNC_DIR:-}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-30000}
KEEP_CKPT=${KEEP_CKPT:-2}
KEEP_BUFFERS=${KEEP_BUFFERS:-10}
SYNC_MIN=${SYNC_MIN:-10}
export HILSERL_SIM_CKPT_PERIOD=${HILSERL_SIM_CKPT_PERIOD:-1000}
export MPLBACKEND=Agg  # Colab's inline backend is not installed in the venv

sync_out() {
  [ -n "$SYNC_DIR" ] || return 0
  mkdir -p "$SYNC_DIR"
  rsync -a --delete --exclude '*tmp*' "$RUN_DIR"/ "$SYNC_DIR"/ && echo "$(date +%T) synced to $SYNC_DIR"
}
latest_ckpt() { ls -d "$RUN_DIR"/checkpoint_* 2>/dev/null | grep -E '/checkpoint_[0-9]+$' | sed 's/.*_//' | sort -n | tail -1; }
env_steps() { [ -f "$RUN_DIR/episodes.csv" ] && awk -F, 'NR>1{s+=$5} END{print s+0}' "$RUN_DIR/episodes.csv" || echo 0; }
# Sort paths by the step number at the end of the name (checkpoint_<n>, transitions_<n>.pkl), oldest first.
by_step() { awk '{n = $0; sub(/\.pkl$/, "", n); sub(/.*_/, "", n); print n "\t" $0}' | sort -n | cut -f2-; }
prune() {
  ls -d "$RUN_DIR"/checkpoint_* 2>/dev/null | grep -E '/checkpoint_[0-9]+$' | by_step | head -n -"$KEEP_CKPT" | xargs -r -d '\n' rm -rf
  ls "$RUN_DIR"/buffer/transitions_*.pkl 2>/dev/null | by_step | head -n -"$KEEP_BUFFERS" | xargs -r -d '\n' rm -f
}

# Restore from SYNC_DIR, and make the run folder something train_rlpd.py can resume from.
if [ ! -d "$RUN_DIR" ] && [ -n "$SYNC_DIR" ] && [ -d "$SYNC_DIR" ]; then
  echo "restoring $RUN_DIR from $SYNC_DIR"; mkdir -p "$RUN_DIR"; rsync -a "$SYNC_DIR"/ "$RUN_DIR"/
fi
if [ -d "$RUN_DIR" ] && [ -z "$(latest_ckpt)" ]; then
  # No checkpoint yet: the learner cannot resume (it would crash on latest_checkpoint=None), so start over.
  aside="$RUN_DIR.no-ckpt-$(date +%Y%m%d-%H%M%S)"
  echo "no checkpoint in $RUN_DIR, moving it to $aside and starting fresh"; mv "$RUN_DIR" "$aside"
fi
if [ -d "$RUN_DIR" ] && ! ls "$RUN_DIR"/buffer/transitions_*.pkl >/dev/null 2>&1; then
  # The actor reads its start step from the newest buffer dump and crashes if there is none.
  mkdir -p "$RUN_DIR/buffer"
  "$WORKDIR/.venv/bin/python" -c "import pickle,sys; pickle.dump([], open(sys.argv[1], 'wb'))" "$RUN_DIR/buffer/transitions_0.pkl"
fi
RESUME=$([ -d "$RUN_DIR" ] && echo "checkpoint $(latest_ckpt), $(env_steps) env steps" || echo fresh)
mkdir -p "$RUN_DIR/logs"
echo "$(date +%T) start ($RESUME), target $TARGET_ENV_STEPS env steps"

cd "$WORKDIR/hil-serl/examples"
source "$WORKDIR/.venv/bin/activate"
unset PYTHONPATH
export CKPT="$RUN_DIR" DEMO_PATH
STAMP=$(date +%Y%m%d-%H%M%S)
LLOG="$RUN_DIR/logs/learner-$STAMP.log"; ALOG="$RUN_DIR/logs/actor-$STAMP.log"
# train_rlpd.py asks "Press Enter to resume" when the checkpoint folder exists, so feed it newlines.
yes "" | XLA_PYTHON_CLIENT_MEM_FRACTION=${LEARNER_FRAC:-.72} bash experiments/sim_pick_cube/run_learner.sh > "$LLOG" 2>&1 &
LP=$!
until grep -q "starting learner loop" "$LLOG" || ! kill -0 $LP 2>/dev/null; do sleep 2; done
kill -0 $LP 2>/dev/null || { echo "learner died, see $LLOG"; tail -5 "$LLOG"; exit 1; }
yes "" | HILSERL_SIM_EPISODE_LOG="$RUN_DIR/episodes.csv" XLA_PYTHON_CLIENT_MEM_FRACTION=${ACTOR_FRAC:-.15} \
  bash experiments/sim_pick_cube/run_actor.sh > "$ALOG" 2>&1 &
AP=$!
echo "$(date +%T) learner and actor running (logs in $RUN_DIR/logs)"

stop() {
  pkill -INT -f "train_rlpd.py --exp_name=sim_pick_cube" 2>/dev/null; sleep 5
  pkill -f "train_rlpd.py --exp_name=sim_pick_cube" 2>/dev/null; sleep 2
  prune; sync_out
}
trap 'echo interrupted; stop; exit 130' INT TERM

# Latest tqdm count in a log ("<n>/<total> ["); pattern selects the progress bar.
tqdm_n() { tail -c 4000 "$1" | tr '\r' '\n' | grep -a "$2" | grep -aoE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1; }
last_sync=$(date +%s); last_ckpt=$(latest_ckpt)
prev_t=$(date +%s); prev_a=0; prev_l=0
while true; do
  sleep 30
  prune
  steps=$(env_steps); ck=$(latest_ckpt)
  sr=$(awk -F, 'NR>1 && $6==0 {n++; s+=$4} END{if(n) printf "%d/%d", s, n; else print "0/0"}' "$RUN_DIR/episodes.csv" 2>/dev/null)
  last20=$(awk -F, 'NR>1 && $6==0 {a[n++]=$4} END{s=0; m=0; for(i=n-20;i<n;i++) if(i>=0){s+=a[i]; m++}; if(m) printf "%d/%d", s, m; else print "-"}' "$RUN_DIR/episodes.csv" 2>/dev/null)
  now=$(date +%s); a=$(tqdm_n "$ALOG" '/'); l=$(tqdm_n "$LLOG" 'learner'); a=${a:-0}; l=${l:-0}
  rates=$(awk -v da=$((a - prev_a)) -v dl=$((l - prev_l)) -v dt=$((now - prev_t)) 'BEGIN{printf "%.1f env steps/s, %.1f learner updates/s", da/dt, dl/dt}')
  prev_t=$now; prev_a=$a; prev_l=$l
  echo "$(date +%T) env steps $steps, $rates, checkpoint ${ck:-none}, policy successes (no takeover): $sr overall, $last20 in last 20"
  if [ "${ck:-}" != "${last_ckpt:-}" ] || [ $(( $(date +%s) - last_sync )) -ge $(( SYNC_MIN * 60 )) ]; then
    sync_out; last_sync=$(date +%s); last_ckpt=$ck
  fi
  kill -0 $LP 2>/dev/null || { echo "learner exited, see $LLOG"; tail -5 "$LLOG"; break; }
  kill -0 $AP 2>/dev/null || { echo "actor exited, see $ALOG"; tail -5 "$ALOG"; break; }
  [ "$steps" -ge "$TARGET_ENV_STEPS" ] && { echo "reached $steps env steps"; break; }
done
stop
echo "$(date +%T) stopped, latest checkpoint $(latest_ckpt)"
