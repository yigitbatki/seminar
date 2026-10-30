#!/usr/bin/env bash
# Sets up HIL-SERL with the simulated sim_pick_cube task.
# Usage:  bash setup.sh [WORKDIR]      (SKIP_INSTALL=1 to only clone, patch and copy)
# Clones rail-berkeley/hil-serl and rail-berkeley/serl (for franka_sim) at pinned commits,
# applies our mappings.py change, copies the experiment in, then runs install.sh.
set -euo pipefail

WORKDIR="${1:-$HOME/hilserl-work}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HILSERL_COMMIT=c32939bccb65f3b8c43a9f9add3d322d4ab0264a
SERL_COMMIT=1fa2af7496be042e43d36120a7c8d1b71f1bfb59
mkdir -p "$WORKDIR"
cd "$WORKDIR"

clone_at() {  # url dir commit
  if [ ! -d "$2" ]; then
    git init -q "$2"
    git -C "$2" remote add origin "$1"
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q FETCH_HEAD
  fi
}
clone_at https://github.com/rail-berkeley/hil-serl.git hil-serl "$HILSERL_COMMIT"
clone_at https://github.com/rail-berkeley/serl.git serl "$SERL_COMMIT"

# Register the sim task; real-robot configs become optional (they need franka_env).
if ! grep -q sim_pick_cube hil-serl/examples/experiments/mappings.py; then
  git -C hil-serl apply "$HERE/mappings.patch"
fi
# Upstream bug: the from-scratch ResNet encoder (encoder_type="resnet") rejects the `encode` argument.
if ! grep -q "encode: bool = True,  # passed by EncodingWrapper" hil-serl/serl_launcher/serl_launcher/vision/resnet_v1.py; then
  git -C hil-serl apply "$HERE/resnet_scratch_encoder.patch"
fi
# Holds the actor back when the learner falls behind, so the update-to-data ratio does not depend on the GPU.
if ! grep -q "HILSERL_SIM_UTD_FILE" hil-serl/examples/train_rlpd.py; then
  git -C hil-serl apply "$HERE/utd_lock.patch"
fi
cp -r "$HERE/experiments/sim_pick_cube" hil-serl/examples/experiments/
cp "$HERE/requirements-hilserl.txt" "$HERE/install.sh" "$HERE/smoke_test.sh" .
echo "hil-serl $(git -C hil-serl rev-parse --short HEAD), serl $(git -C serl rev-parse --short HEAD), experiment copied to hil-serl/examples/experiments/sim_pick_cube"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  bash install.sh "$WORKDIR"
fi
