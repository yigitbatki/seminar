#!/usr/bin/env bash
# Resumable ConRFT training on sim_pick_cube, one stage at a time.
# Re-running it with the same RUN_DIR (or SYNC_DIR) continues from the latest checkpoint.
#
# usage: STAGE=pretrain|online DEMO_PATH=... RUN_DIR=... [SYNC_DIR=...] bash run_conrft.sh
#   STAGE=pretrain    offline training on the demos (learner only) until PRETRAIN_STEPS.
#                     Cal-ConRFT: Q_WEIGHT=0.1 BC_WEIGHT=1.0; SFT: Q_WEIGHT=0 BC_WEIGHT=1.0
#   STAGE=online      learner + actor with the scripted expert, from INIT_CKPT (a pretrain
#                     run's checkpoint_<N> folder) until TARGET_ENV_STEPS env steps.
#                     HIL-ConRFT: Q_WEIGHT=1.0 BC_WEIGHT=0.1; HG-DAgger: Q_WEIGHT=0 BC_WEIGHT=1 BC_DEMO_ONLY=1
#   WORKDIR           where setup.sh put conrft and .venv (default ~/conrft-sim-work)
#   RUN_DIR           run folder on fast local disk: RUN_DIR/ckpt is train_conrft_octo.py's
#                     checkpoint_path, RUN_DIR/logs holds the logs and episodes.csv
#   SYNC_DIR          optional copy of RUN_DIR (e.g. Google Drive), refreshed every SYNC_MIN minutes
#                     and after each new checkpoint; used to restore RUN_DIR when it is missing
#   KEEP_CKPT=2       checkpoints kept
#   KEEP_BUFFERS=10   actor buffer dumps kept (online; only for resuming)
#   LEARNER_FRAC, ACTOR_FRAC   GPU memory caps (fractions of the card)
#   CONRFT_SIM_CKPT_PERIOD     learner steps between checkpoints
#   HILSERL_SIM_UTD=1          online: learner updates per env step the actor waits for (0 = no lock)
#   HILSERL_SIM_UTD_SLACK=500  online: env steps the actor may be ahead of that ratio; keep it above
#                              training_starts (100), or the actor stops before the learner starts
set -u
WORKDIR=${WORKDIR:-$HOME/conrft-sim-work}
STAGE=${STAGE:?set STAGE to pretrain or online}
: "${DEMO_PATH:?set DEMO_PATH to the demo .pkl}"
: "${RUN_DIR:?set RUN_DIR}"
SYNC_DIR=${SYNC_DIR:-}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-20000}
TARGET_ENV_STEPS=${TARGET_ENV_STEPS:-30000}
KEEP_CKPT=${KEEP_CKPT:-2}
KEEP_BUFFERS=${KEEP_BUFFERS:-10}
SYNC_MIN=${SYNC_MIN:-10}
export CONRFT_SIM_CKPT_PERIOD=${CONRFT_SIM_CKPT_PERIOD:-1000}
export MPLBACKEND=Agg  # Colab's inline backend is not installed in the venv
# train_conrft_octo.py resumes whenever checkpoint_path exists, so nothing else may be created inside it.
CK="$RUN_DIR/ckpt"
LOGDIR="$RUN_DIR/logs"
EPLOG="$LOGDIR/episodes.csv"

sync_out() {
  [ -n "$SYNC_DIR" ] || return 0
  mkdir -p "$SYNC_DIR"
  # Files the notebook or the queue add next to the run (results, videos) are excluded, so --delete keeps them.
  rsync -a --delete --exclude '*tmp*' --exclude 'videos/' --exclude 'results*.json' --exclude 'settings.json' \
    --exclude 'DONE' "$RUN_DIR"/ "$SYNC_DIR"/ && echo "$(date +%T) synced to $SYNC_DIR"
}
latest_ckpt() { ls -d "$CK"/checkpoint_* 2>/dev/null | grep -E '/checkpoint_[0-9]+$' | sed 's/.*_//' | sort -n | tail -1; }
env_steps() { [ -f "$EPLOG" ] && awk -F, '{sub(/\r$/, "")} NR>1{s+=$5} END{print s+0}' "$EPLOG" || echo 0; }
# Sort paths by the step number at the end of the name (checkpoint_<n>, transitions_<n>.pkl), oldest first.
by_step() { awk '{n = $0; sub(/\.pkl$/, "", n); sub(/.*_/, "", n); print n "\t" $0}' | sort -n | cut -f2-; }
prune() {
  ls -d "$CK"/checkpoint_* 2>/dev/null | grep -E '/checkpoint_[0-9]+$' | by_step | head -n -"$KEEP_CKPT" | xargs -r -d '\n' rm -rf
  ls "$CK"/buffer/transitions_*.pkl 2>/dev/null | by_step | head -n -"$KEEP_BUFFERS" | xargs -r -d '\n' rm -f
}
# Latest tqdm count in a log ("<n>/<total> ["); pattern selects the progress bar.
tqdm_n() { tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' | grep -a "$2" | grep -aoE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1; }

# Restore from SYNC_DIR, and make the run folder something train_conrft_octo.py can resume from.
if [ ! -d "$RUN_DIR" ] && [ -n "$SYNC_DIR" ] && [ -d "$SYNC_DIR/ckpt" ]; then
  echo "restoring $RUN_DIR from $SYNC_DIR"; mkdir -p "$RUN_DIR"
  rsync -a --exclude 'videos/' --exclude 'results*.json' --exclude 'settings.json' --exclude 'DONE' "$SYNC_DIR"/ "$RUN_DIR"/
fi
if [ -d "$RUN_DIR" ] && [ -z "$(latest_ckpt)" ]; then
  # No checkpoint yet: the learner cannot resume (it would crash on latest_checkpoint=None), so start over.
  aside="$RUN_DIR.no-ckpt-$(date +%Y%m%d-%H%M%S)"
  echo "no checkpoint in $RUN_DIR, moving it to $aside and starting fresh"; mv "$RUN_DIR" "$aside"
fi
if [ "$STAGE" = online ] && [ ! -d "$CK" ]; then
  : "${INIT_CKPT:?set INIT_CKPT to the pretrain checkpoint_<N> folder to start online training from}"
  mkdir -p "$CK"; cp -r "$INIT_CKPT" "$CK/"
fi
if [ "$STAGE" = online ]; then
  # The learner skips pretraining when its latest checkpoint is at or past --pretrain_steps.
  INIT_STEP=$(ls -d "$CK"/checkpoint_* | grep -E '/checkpoint_[0-9]+$' | sed 's/.*_//' | sort -n | head -1)
  PRETRAIN_STEPS=${INIT_STEP}
  # The update-to-data lock counts online learner steps only. INIT_STEP cannot serve as the origin:
  # prune removes the init checkpoint, so after a resume it is a later checkpoint.
  [ -f "$RUN_DIR/online_init_step" ] || echo "${INIT_CKPT:-_$INIT_STEP}" | sed 's/.*_//' > "$RUN_DIR/online_init_step"
  export HILSERL_SIM_UTD_FILE="$RUN_DIR/learner_step"
  export HILSERL_SIM_UTD=${HILSERL_SIM_UTD:-1}
  export HILSERL_SIM_UTD_OFFSET=$(cat "$RUN_DIR/online_init_step")
fi
RESUME=$([ -d "$CK" ] && echo "checkpoint $(latest_ckpt), $(env_steps) env steps" || echo fresh)
mkdir -p "$LOGDIR"
echo "$(date +%T) $STAGE stage start ($RESUME), target $([ "$STAGE" = pretrain ] && echo "$PRETRAIN_STEPS pretrain steps" || echo "$TARGET_ENV_STEPS env steps")"
if [ "$STAGE" = pretrain ] && [ "$(latest_ckpt)" = "$PRETRAIN_STEPS" ]; then
  echo "pretraining already finished"; sync_out; exit 0
fi

cd "$WORKDIR/conrft/examples"
source "$WORKDIR/.venv/bin/activate"
unset PYTHONPATH
export CKPT="$CK" DEMO_PATH PRETRAIN_STEPS
STAMP=$(date +%Y%m%d-%H%M%S)
LLOG="$LOGDIR/learner-$STAGE-$STAMP.log"; ALOG="$LOGDIR/actor-$STAMP.log"
if [ "$STAGE" = pretrain ]; then
  XLA_PYTHON_CLIENT_MEM_FRACTION=${LEARNER_FRAC:-.5} bash experiments/sim_pick_cube/run_learner_conrft_pretrain.sh > "$LLOG" 2>&1 &
else
  # train_conrft_octo.py may ask "Press Enter to resume"; feed it newlines.
  yes "" | XLA_PYTHON_CLIENT_MEM_FRACTION=${LEARNER_FRAC:-.5} bash experiments/sim_pick_cube/run_learner_conrft.sh > "$LLOG" 2>&1 &
fi
LP=$!
AP=""
if [ "$STAGE" = online ]; then
  until grep -q "starting learner loop" "$LLOG" || ! kill -0 $LP 2>/dev/null; do sleep 2; done
  kill -0 $LP 2>/dev/null || { echo "learner died, see $LLOG"; tail -20 "$LLOG"; exit 1; }
  HILSERL_SIM_EPISODE_LOG="$EPLOG" XLA_PYTHON_CLIENT_MEM_FRACTION=${ACTOR_FRAC:-.2} \
    bash experiments/sim_pick_cube/run_actor_conrft.sh > "$ALOG" 2>&1 &
  AP=$!
fi
echo "$(date +%T) $STAGE running (logs in $LOGDIR)"

stop() {
  pkill -INT -f "train_conrft_octo.py --exp_name=sim_pick_cube" 2>/dev/null; sleep 5
  pkill -f "train_conrft_octo.py --exp_name=sim_pick_cube" 2>/dev/null; sleep 2
  prune; sync_out
}
trap 'echo interrupted; stop; exit 130' INT TERM

last_sync=$(date +%s); last_ckpt=$(latest_ckpt)
prev_t=$(date +%s); prev_a=0; prev_l=0
status=0
while true; do
  sleep 30
  prune
  ck=$(latest_ckpt)
  now=$(date +%s)
  if [ "$STAGE" = pretrain ]; then
    l=$(tqdm_n "$LLOG" 'pretraining'); l=${l:-0}
    rate=$(awk -v dl=$((l - prev_l)) -v dt=$((now - prev_t)) 'BEGIN{printf "%.2f", dl/dt}')
    echo "$(date +%T) pretrain step $l/$PRETRAIN_STEPS, $rate steps/s, checkpoint ${ck:-none}"
  else
    steps=$(env_steps)
    sr=$(awk -F, '{sub(/\r$/, "")} NR>1 && $6+0==0 {n++; s+=$4} END{if(n) printf "%d/%d", s, n; else print "0/0"}' "$EPLOG" 2>/dev/null)
    last20=$(awk -F, '{sub(/\r$/, "")} NR>1 && $6+0==0 {a[n++]=$4} END{s=0; m=0; for(i=n-20;i<n;i++) if(i>=0){s+=a[i]; m++}; if(m) printf "%d/%d", s, m; else print "-"}' "$EPLOG" 2>/dev/null)
    a=$(tqdm_n "$ALOG" '/'); l=$(tqdm_n "$LLOG" 'learner'); a=${a:-0}; l=${l:-0}
    rates=$(awk -v da=$((a - prev_a)) -v dl=$((l - prev_l)) -v dt=$((now - prev_t)) 'BEGIN{printf "%.1f env steps/s, %.1f learner updates/s", da/dt, dl/dt}')
    prev_a=$a
    echo "$(date +%T) env steps $steps, $rates, checkpoint ${ck:-none}, policy successes (no takeover): $sr overall, $last20 in last 20"
  fi
  prev_t=$now; prev_l=$l
  if [ "${ck:-}" != "${last_ckpt:-}" ] || [ $(( $(date +%s) - last_sync )) -ge $(( SYNC_MIN * 60 )) ]; then
    sync_out; last_sync=$(date +%s); last_ckpt=$ck
  fi
  if ! kill -0 $LP 2>/dev/null; then
    if [ "$STAGE" = pretrain ] && [ "$(latest_ckpt)" = "$PRETRAIN_STEPS" ]; then
      echo "pretraining finished at step $PRETRAIN_STEPS"
    else
      echo "learner exited, see $LLOG"; tail -20 "$LLOG"; status=1
    fi
    break
  fi
  if [ "$STAGE" = online ]; then
    kill -0 $AP 2>/dev/null || { echo "actor exited, see $ALOG"; tail -20 "$ALOG"; status=1; break; }
    [ "$steps" -ge "$TARGET_ENV_STEPS" ] && { echo "reached $steps env steps"; break; }
  fi
done
stop
echo "$(date +%T) stopped, latest checkpoint $(latest_ckpt)"
exit $status
