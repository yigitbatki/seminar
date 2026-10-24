#!/usr/bin/env bash
# Sets up ConRFT + Octo + SERL's franka_sim in a Python 3.10 environment.
# Usage:  bash setup.sh [gpu|cpu] [WORKDIR]      (SKIP_INSTALL=1 to only clone, patch and copy)
# On Colab: run from a GPU runtime with "gpu". Requires `uv` (installed below if missing).
# The simulated task (sim_env.py) is shared with the HIL-SERL setup in ../hilserl.
set -euo pipefail

DEVICE="${1:-gpu}"
WORKDIR="${2:-$HOME/conrft-sim-work}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# 1. Code at pinned commits: ConRFT, its Octo fork, and SERL (for the franka_sim simulator).
clone_at() {  # url dir commit
  if [ ! -d "$2" ]; then
    git init -q "$2"
    git -C "$2" remote add origin "$1"
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q FETCH_HEAD
  fi
}
clone_at https://github.com/cccedric/conrft.git conrft a779fde7fa5db5a469960a8490c100f35b41b49e
clone_at https://github.com/cccedric/octo.git octo e454c7a1939163a7e1c70670d34dc504c4825247
clone_at https://github.com/rail-berkeley/serl.git serl 1fa2af7496be042e43d36120a7c8d1b71f1bfb59

# 2. Fixes in the public ConRFT code (see apply_patches.py).
python3 "$HERE/apply_patches.py" conrft

# 3. Add the simulated task to ConRFT's experiment registry.
cp -r "$HERE/experiments/sim_pick_cube" conrft/examples/experiments/
cp "$HERE/../hilserl/experiments/sim_pick_cube/sim_env.py" conrft/examples/experiments/sim_pick_cube/
cp "$HERE/mappings.py" conrft/examples/experiments/mappings.py
cp "$HERE/test_sim_env.py" "$HERE/tools/tiny_octo_smoke.py" conrft/examples/
cp "$HERE"/run_*.sh conrft/examples/experiments/sim_pick_cube/
echo "conrft $(git -C conrft rev-parse --short HEAD), octo $(git -C octo rev-parse --short HEAD), serl $(git -C serl rev-parse --short HEAD); sim_pick_cube copied"
[ "${SKIP_INSTALL:-0}" = "1" ] && exit 0

# 4. Python 3.10 environment (Octo needs TensorFlow 2.15, which has no wheels for Python 3.12).
# Colab points uv at the system Python (3.12+); always target our 3.10 venv instead.
unset UV_SYSTEM_PYTHON UV_PYTHON
[ -d .venv ] || uv venv -p 3.10 .venv
export VIRTUAL_ENV="$WORKDIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [ "$DEVICE" = "gpu" ]; then
  JAX_SPEC='jax[cuda11_pip]==0.4.20'
  JAX_INDEX=(-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html)
else
  JAX_SPEC='jax[cpu]==0.4.20'
  JAX_INDEX=()
fi

# 5. One pinned set that satisfies Octo, ConRFT's serl_launcher and franka_sim together.
uv pip install --python "$WORKDIR/.venv/bin/python" "${JAX_INDEX[@]}" "$JAX_SPEC" -r "$HERE/requirements-pinned.txt"
# jaxlib 0.4.20+cuda11.cudnn86 needs cuDNN 8.x; left unpinned, uv picks cuDNN 9 and jax falls back to CPU.
if [ "$DEVICE" = "gpu" ]; then uv pip install --python "$WORKDIR/.venv/bin/python" "nvidia-cudnn-cu11==8.9.6.50"; fi
uv pip install --python "$WORKDIR/.venv/bin/python" --no-deps -e octo -e conrft/serl_launcher -e serl/franka_sim

env -u PYTHONPATH python -c "import jax; print(jax.devices())"

echo
echo "Done. Activate with:  source $WORKDIR/.venv/bin/activate"
echo "Then:                 cd $WORKDIR/conrft/examples && MUJOCO_GL=egl python test_sim_env.py"
