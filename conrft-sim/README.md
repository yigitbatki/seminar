# ConRFT in simulation

This folder runs ConRFT ([paper](https://arxiv.org/abs/2502.05450), [code](https://github.com/cccedric/conrft)) on a simulated Franka arm lifting a cube, instead of on the real robot. The simulator is the MuJoCo `franka_sim` environment from [SERL](https://github.com/rail-berkeley/serl). A scripted expert takes the place of the human with the SpaceMouse: it records the demos and makes the online corrections.

The task file (`sim_env.py`) is shared with the HIL-SERL setup in `../hilserl`, so both methods get the same task, cube positions (from the same seed), scripted expert, interventions, 10 Hz control rate and evaluation cubes, and their results end up in one table.

## What is in here
| File | What it does |
|---|---|
| `experiments/sim_pick_cube/config.py` | ConRFT's experiment config on the shared `sim_env.py`: 256 px front camera (`side_policy_256`), 128 px wrist camera (`wrist_1`), 2 stacked frames, reward +10 on success / −0.05 per step, Octo path, hyperparameters |
| `apply_patches.py` | Fixes in the public ConRFT code (two broken imports, an extra GPU copy of the agent) and two small additions: `--bc_demo_only` for the HG-DAgger baseline and a guaranteed final pretraining checkpoint |
| `setup.sh`, `requirements-pinned.txt` | Clones ConRFT, its Octo fork and SERL at pinned commits, patches them, copies the task in, and installs a Python 3.10 environment |
| `run_conrft.sh` | Resumable training of one stage (`STAGE=pretrain` or `online`), with checkpoints copied to a sync folder (e.g. Google Drive) |
| `run_*.sh` | The underlying ConRFT commands: demo recording, offline pretraining, online learner and actor |
| `evaluate_conrft.py` | Evaluation on the same fixed cubes and in the same results format as `../hilserl/evaluate.py` |
| `octo_test.sh` | A short run with the real Octo-small that prints peak GPU memory and speeds |
| `ConRFT_sim_colab.ipynb` | The whole pipeline on Colab, as a resumable experiment queue (uses `../hilserl/experiment_queue.py`) |
| `test_sim_env.py` | Checks observation shapes and the expert's success rate (no Octo needed) |
| `tools/tiny_octo_smoke.py` | Runs any ConRFT script with a tiny random Octo, for plumbing tests without the real weights (`CONRFT_TINY_OCTO=1`) |

## Stages
- **SFT** (`STAGE=pretrain Q_WEIGHT=0 BC_WEIGHT=1`): offline behaviour cloning of Octo's consistency-policy head on the demos.
- **Cal-ConRFT** (`STAGE=pretrain`, Q 0.1, BC 1.0): offline Cal-QL + BC on the demos, 20k steps as in the paper.
- **HIL-ConRFT** (`STAGE=online`, Q 1.0, BC 0.1): online with the scripted expert's interventions, from a pretraining checkpoint (`INIT_CKPT`), half of each batch from demos + interventions.
- **HG-DAgger** (`STAGE=online Q_WEIGHT=0 BC_WEIGHT=1 BC_DEMO_ONLY=1`): online, but only imitating the demos and interventions.

## Running it
On Colab: open `ConRFT_sim_colab.ipynb` from GitHub on an A100 (or L4) runtime with High RAM and run the cells in order; the HIL-SERL demos must already be in `MyDrive/hilserl-runs/demos` (ConRFT records its own demos at the same 20 cube positions).

On a Linux machine with an NVIDIA GPU:
```bash
bash setup.sh gpu ~/conrft-sim-work
source ~/conrft-sim-work/.venv/bin/activate
cd ~/conrft-sim-work/conrft/examples
MUJOCO_GL=egl python test_sim_env.py --episodes 5
N_DEMOS=20 bash experiments/sim_pick_cube/run_record_demos.sh
export DEMO_PATH=$(ls $PWD/demo_data/sim_pick_cube_20_demos_*.pkl | tail -1)
STAGE=pretrain RUN_DIR=~/runs/cal bash ~/seminar-repo/conrft-sim/run_conrft.sh
STAGE=online RUN_DIR=~/runs/hil INIT_CKPT=~/runs/cal/ckpt/checkpoint_20000 bash ~/seminar-repo/conrft-sim/run_conrft.sh
python ~/seminar-repo/conrft-sim/evaluate_conrft.py --ckpt_dir ~/runs/hil/ckpt --episodes 100
```

## Things to know
- **GPU memory:** the actor's training state includes Octo's T5-base text encoder (about 110M parameters) plus optimizer state, and online training holds two copies (learner and actor). 6 GB is not enough even at batch size 32; use an A100 (40 GB) or check an L4 (24 GB) with `octo_test.sh` first.
- **Evaluation:** ConRFT's consistency policy always samples (its `sample_actions` has no mode), so only sampled-action success rates are reported.
- **Memory per transition:** about 0.25 MB (256 px and 128 px images, 2 frames), so the replay buffer defaults to 10k transitions (`CONRFT_SIM_BUFFER`). A 20-demo file is about 1.2 GB because it also stores Octo embeddings.
- **Settings:** task settings use the same `HILSERL_SIM_*` environment variables as HIL-SERL (see `experiments/sim_pick_cube/config.py`); ConRFT's own are `CONRFT_SIM_BUFFER`, `CONRFT_SIM_BATCH_SIZE`, `CONRFT_SIM_CKPT_PERIOD` and `CONRFT_OCTO_PATH`.
- **Other fixes in this setup:** TensorFlow 2.15 needs `protobuf<4`, so `tensorflow-metadata` is pinned to 1.14.0 and `wandb` to 0.16.6; jaxlib 0.4.20+cuda11 needs cuDNN 8.x (`nvidia-cudnn-cu11==8.9.6.50`); ConRFT's agent asserts 7-D actions, so the simulator uses that layout.
