#!/usr/bin/env bash
# Sets up ConRFT + Octo + SERL's franka_sim in a Python 3.10 environment.
# Usage:  bash setup.sh [gpu|cpu] [WORKDIR]
# On Colab: run from a GPU runtime with "gpu". Requires `uv` (installed below if missing).
set -euo pipefail

DEVICE="${1:-gpu}"
WORKDIR="${2:-$HOME/conrft-sim-work}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# 1. Code: ConRFT, its Octo fork, and SERL (for the franka_sim simulator).
[ -d conrft ] || git clone --depth 1 https://github.com/cccedric/conrft.git
[ -d octo ]   || git clone --depth 1 https://github.com/cccedric/octo.git
[ -d serl ]   || git clone --depth 1 https://github.com/rail-berkeley/serl.git

# 2. Python 3.10 environment (Octo needs TensorFlow 2.15, which has no wheels for Python 3.12).
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

# 3. One pinned set that satisfies Octo, ConRFT's serl_launcher and franka_sim together.
uv pip install --python "$WORKDIR/.venv/bin/python" "${JAX_INDEX[@]}" "$JAX_SPEC" -r "$HERE/requirements-pinned.txt"
# jaxlib 0.4.20+cuda11.cudnn86 needs cuDNN 8.x; left unpinned, uv picks cuDNN 9 and jax falls back to CPU.
if [ "$DEVICE" = "gpu" ]; then uv pip install --python "$WORKDIR/.venv/bin/python" "nvidia-cudnn-cu11==8.9.6.50"; fi
uv pip install --python "$WORKDIR/.venv/bin/python" --no-deps -e octo -e conrft/serl_launcher -e serl/franka_sim

# 4. Fix two import bugs in the public ConRFT code.
python "$HERE/apply_patches.py" conrft

# 5. Add the simulated task to ConRFT's experiment registry.
cp -r "$HERE/experiments/sim_pick_cube" conrft/examples/experiments/
cp "$HERE/mappings.py" conrft/examples/experiments/mappings.py
cp "$HERE/test_sim_env.py" conrft/examples/
cp "$HERE"/run_*.sh conrft/examples/experiments/sim_pick_cube/

echo
echo "Done. Activate with:  source $WORKDIR/.venv/bin/activate"
echo "Then:                 cd $WORKDIR/conrft/examples && MUJOCO_GL=egl python test_sim_env.py"
