#!/usr/bin/env bash
# Python 3.10 venv with jax 0.4.20 (CUDA 11) and HIL-SERL. Usage: bash install.sh [WORKDIR]
# Uses the CUDA 11 stack instead of the README's jax[cuda12_pip]==0.4.35; hil-serl uses no newer JAX APIs.
set -euo pipefail
WORKDIR="${1:-$HOME/hilserl-work}"
cd "$WORKDIR"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH" UV_HTTP_TIMEOUT=600
[ -d .venv ] || uv venv -p 3.10 .venv
export VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH
uv pip install -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 'jax[cuda11_pip]==0.4.20' -r requirements-hilserl.txt
uv pip install --no-deps -e hil-serl/serl_launcher -e serl/franka_sim
env -u PYTHONPATH python -c "import jax; print(jax.devices())"
