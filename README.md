# Simulated HIL-SERL and ConRFT on a Franka cube-lift task

Code bonus for the TUM seminar on Robotics Science and Systems Intelligence. It reproduces two
human-in-the-loop RL methods for robot manipulation in simulation instead of on a real Franka arm:

- **HIL-SERL**: Luo et al., *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning* (2024). [paper](https://arxiv.org/abs/2410.21845), [code](https://github.com/rail-berkeley/hil-serl)
- **ConRFT**: Chen et al., *ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy* (2025). [paper](https://arxiv.org/abs/2502.05450), [code](https://github.com/cccedric/conrft)

The task is SERL's MuJoCo `franka_sim` PandaPickCube: lift a cube by 10 cm from camera images
(front and wrist) plus gripper state, with a sparse success reward. A scripted expert with privileged
state replaces the human with the SpaceMouse: it records the demos and takes over during online
training (`info["intervene_action"]`, in a configurable fraction of episodes).

This repository holds only our files. The setup scripts clone the upstream code at pinned commits and
copy our task in, so nothing upstream is vendored here.

## Layout
| Path | What it is |
|---|---|
| `hilserl/setup.sh` | Clones `rail-berkeley/hil-serl` @ `c32939b` and `rail-berkeley/serl` @ `1fa2af7` (for `franka_sim`), applies `mappings.patch`, copies the task in, runs `install.sh` |
| `hilserl/experiments/sim_pick_cube/` | The task for HIL-SERL: `sim_env.py` (env, scripted expert, intervention and gripper-penalty wrappers, episode CSV log), `config.py`, `run_*.sh` |
| `hilserl/mappings.patch` | Registers `sim_pick_cube`; makes the real-robot configs optional (they need `franka_env`) |
| `hilserl/install.sh`, `requirements-hilserl.txt` | Python 3.10 venv: jax 0.4.20 (CUDA 11), cuDNN 8.9.6.50, HIL-SERL's `serl_launcher`, `franka_sim` |
| `hilserl/smoke_test.sh` | Runs learner + actor together and stops after N env steps or N minutes |
| `conrft-sim/` | The task for ConRFT (Octo-small backbone) with `setup.sh`, import fixes (`apply_patches.py`), run scripts and a Colab notebook. See `conrft-sim/README.md` |

## Quickstart: HIL-SERL
```bash
bash hilserl/setup.sh ~/hilserl-work            # clone, patch, copy, install
cd ~/hilserl-work/hil-serl/examples
source ~/hilserl-work/.venv/bin/activate && unset PYTHONPATH
N_DEMOS=20 bash experiments/sim_pick_cube/run_record_demos.sh
export DEMO_PATH=$(ls $PWD/demo_data/sim_pick_cube_*.pkl | tail -1)
# short smoke test (learner + actor on one GPU):
MAX_ENV_STEPS=2000 MAX_MIN=20 bash ~/hilserl-work/smoke_test.sh smoke1
# or start them yourself in two shells:
XLA_PYTHON_CLIENT_MEM_FRACTION=.72 bash experiments/sim_pick_cube/run_learner.sh
XLA_PYTHON_CLIENT_MEM_FRACTION=.15 bash experiments/sim_pick_cube/run_actor.sh
```
Task settings: 2 cameras at 128×128 (`front`, `wrist`), 7-D actions with a learned gripper
(`single-arm-learned-gripper`, grasp DQN critic), reward 1 on success and 0 otherwise, episode ends on
success or after 100 steps, frozen pretrained ResNet-10 encoder, RLPD with 50/50 demo/online batches.
Environment variables: `HILSERL_SIM_MODE` (`demo`/`online`), `HILSERL_SIM_INTERVENE_PROB` (0.2),
`HILSERL_SIM_BUFFER` (20000), `HILSERL_SIM_BATCH_SIZE` (256), `HILSERL_SIM_CKPT_PERIOD` (2000),
`HILSERL_SIM_EPISODE_LOG` (CSV path).

## Quickstart: ConRFT
```bash
bash conrft-sim/setup.sh gpu ~/conrft-sim-work
source ~/conrft-sim-work/.venv/bin/activate && cd ~/conrft-sim-work/conrft/examples
MUJOCO_GL=egl python test_sim_env.py --episodes 5
bash experiments/sim_pick_cube/run_record_demos.sh
export DEMO_PATH=$(ls $PWD/demo_data/sim_pick_cube_*.pkl | tail -1)
bash experiments/sim_pick_cube/run_learner_conrft_pretrain.sh    # needs > 6 GB GPU memory, see below
```
The Colab notebook `conrft-sim/ConRFT_sim_colab.ipynb` expects this folder in Drive as `MyDrive/conrft-sim`.

## Tested on
RTX 3060 Laptop GPU (6 GB), 16 CPU threads, Windows + WSL2 (Ubuntu, 7 GB RAM), driver 546.80, 24 Oct 2026.

| Step | HIL-SERL | ConRFT |
|---|---|---|
| GPU visible to jax | yes | yes |
| Scripted expert | 3/3 test episodes | 5/5 episodes, mean 60.6 steps |
| 20 demos | 5.5 min, 236 MB | 10.4 min, 1.19 GB (256 px images + Octo embeddings) |
| Training | learner + actor ran 2,092 env steps in 13.4 min; checkpoints saved | pretrain out of GPU memory (see below) |
| Speed | 2.7–2.8 env steps/s, ~2.7 learner updates/s (GPU ~99% busy) | — |
| Peak memory | 5.55 GB GPU (both processes), 5.2 GB RAM | — |

After 2k env steps the HIL-SERL policy had not succeeded on its own yet (0/17 non-intervened
episodes; all 6 expert takeovers succeeded), which is expected that early. At this speed 30k env steps
take about 3 h and 50k about 5 h.

## Known limits
- **ConRFT pretraining does not fit in 6 GB of GPU memory**, even at batch size 32. The Octo agent's
  train state carries the T5-base text encoder (its 32128×768 embedding shows up repeatedly in XLA's
  peak buffers), so memory is dominated by parameters and optimizer state, not by the batch. Use a GPU
  with more memory (for example Colab L4/A100).
- **Rendering under WSL2 is slow:** MuJoCo's EGL backend goes through WSL's D3D12 layer and takes about
  90 ms per 128×128 camera frame (physics takes 0.5 ms), which caps the actor at about 3 env steps/s.
  Mesa's llvmpipe is slower (~150 ms/frame). Native Linux EGL should be much faster.
- **GPU memory fractions:** with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, the fraction is a hard cap. On a
  6 GB card the HIL-SERL learner needs `.72` at batch 256 (`.5` runs out of memory on the first update)
  and the actor `.15`. All `run_*.sh` scripts read `XLA_PYTHON_CLIENT_MEM_FRACTION` from the environment.
- **cuDNN:** jax 0.4.20+cuda11 needs cuDNN 8.x. Without the `nvidia-cudnn-cu11==8.9.6.50` pin, uv picks
  cuDNN 9 and jax silently falls back to the CPU.
- **RAM:** the HIL-SERL replay buffer holds about 0.1 MB per transition, so the default 20k capacity adds
  about 2 GB when full. With 7 GB in WSL2 use `HILSERL_SIM_BUFFER=10000` or raise the WSL memory limit.
- If your shell sets `PYTHONPATH` (for example from ROS), unset it before running the venv's Python.
