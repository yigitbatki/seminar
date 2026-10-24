# ConRFT in simulation

This folder runs ConRFT ([paper](https://arxiv.org/abs/2502.05450), [code](https://github.com/cccedric/conrft)) on a simulated Franka arm lifting a cube, instead of on the real robot. The simulator is the MuJoCo `franka_sim` environment from [SERL](https://github.com/rail-berkeley/serl). A scripted expert takes the place of the human with the SpaceMouse: it records the demos and makes the online corrections.

The upstream ConRFT scripts (`record_demos_octo.py` and `train_conrft_octo.py`) run unchanged. This folder only adds a new experiment, `sim_pick_cube`, and fixes two import bugs in the public code.

## What is in here
| File | What it does |
|---|---|
| `experiments/sim_pick_cube/sim_env.py` | The simulated task. It also contains the scripted expert, the intervention wrapper that stands in for the SpaceMouse, and the gripper penalty |
| `experiments/sim_pick_cube/config.py` | The ConRFT experiment config: image keys, proprioceptive keys, Octo path, hyperparameters |
| `mappings.py` | Registers `sim_pick_cube` with ConRFT |
| `apply_patches.py` | Fixes two broken imports in the ConRFT repository |
| `setup.sh`, `requirements-pinned.txt` | Python 3.10 environment with versions that work together |
| `run_*.sh` | Demo recording, offline pretraining, online learner and online actor |
| `test_sim_env.py` | Checks observation shapes and the expert's success rate (no Octo needed) |
| `tools/tiny_octo_smoke.py` | Runs any ConRFT script with a tiny random Octo, for offline plumbing tests |
| `ConRFT_sim_colab.ipynb` | The whole pipeline on a Colab GPU |

## Task
- **Observations:** `side_policy_256` (front camera, 256×256), `wrist_1` (wrist camera, 128×128) and `state` (gripper opening, gripper position and gripper velocity), stacked over 2 time steps.
- **Actions:** 7-D, the same layout as the real robot: $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g]$. Rotation is ignored, because the simulated gripper keeps a fixed orientation. $g>0.5$ closes the gripper and $g<-0.5$ opens it.
- **Reward:** sparse, as in the paper. It is $+10$ when the cube is lifted by 10 cm, which also ends the episode, and $-0.05$ per step otherwise. Episodes last at most 100 steps.
- **Simulated human:** `CONRFT_SIM_MODE=demo` lets the expert control every step. `CONRFT_SIM_MODE=online` lets it take over from a random step in a fraction `CONRFT_SIM_INTERVENE_PROB` of episodes (default 0.2). With a little action noise, the expert succeeds in 29 out of 30 episodes, taking about 60 steps.

## Running it
On Colab, copy this folder to `MyDrive/conrft-sim` in Google Drive (or keep `conrft-sim.zip` ready to upload), open `ConRFT_sim_colab.ipynb` on a GPU runtime and run the cells in order. The notebook checks the GPU and the Octo weights first, then records demos, pretrains, runs online fine-tuning and evaluates.

On a Linux machine with an NVIDIA GPU:
```bash
bash setup.sh gpu ~/conrft-sim-work
source ~/conrft-sim-work/.venv/bin/activate
cd ~/conrft-sim-work/conrft/examples
MUJOCO_GL=egl python test_sim_env.py --episodes 5

bash experiments/sim_pick_cube/run_record_demos.sh            # N_DEMOS=20 by default
export DEMO_PATH=$(ls $PWD/demo_data/sim_pick_cube_*.pkl | tail -1)
bash experiments/sim_pick_cube/run_learner_conrft_pretrain.sh  # stage 1: Cal-ConRFT, exits when done
bash experiments/sim_pick_cube/run_learner_conrft.sh &         # stage 2: HIL-ConRFT learner
bash experiments/sim_pick_cube/run_actor_conrft.sh             # stage 2: actor with the simulated human
```

To evaluate a checkpoint, run `run_actor_conrft.sh --eval_checkpoint_step=<step> --eval_n_trajs=20`.

## What I tested, and what I could not
I tested this on 20 Oct 2026 in a CPU-only container that had **no access to Hugging Face**, so I could not download the real Octo-small weights. What I checked:
- The full environment installs with `setup.sh cpu`.
- The simulator produces ConRFT's expected observation shapes: `(2, 256, 256, 3)`, `(2, 128, 128, 3)` and `(2, 7)`.
- The scripted expert succeeds in 29 out of 30 episodes.
- ConRFT's own `record_demos_octo.py` records demos, using a tiny random stand-in Octo (`tools/tiny_octo_smoke.py`).
- ConRFT's offline pretraining stage (`update_calql`) runs end to end and saves checkpoints.
- The online stage (HIL-ConRFT): the learner loads the pretrained checkpoint, fills its buffer from the actor, sends the network to the actor and keeps training while the actor runs with simulated interventions.

**Not tested:** the GPU install (`jax[cuda11_pip]==0.4.20` on Colab) and the real `hf://rail-berkeley/octo-small` weights. Those are the first two things to check on Colab. If loading fails, try `CONRFT_OCTO_PATH=hf://rail-berkeley/octo-small-1.5`.

## Things to know
- **Memory:** each replay slot stores one 256×256 and one 128×128 image. The upstream capacity of 200k slots would need about 45 GB per buffer, so the default here is `CONRFT_SIM_BUFFER=10000` (about 2.3 GB per buffer, and there are two buffers). Demo files are large too: about 60 MB per demo, because they also store Octo embeddings and both observation frames.
- **Speed:** rendering two cameras is slow on a CPU (about 3 steps/s). On a GPU with `MUJOCO_GL=egl` it is much faster.
- **Bugs fixed in the public ConRFT code:** `serl_launcher/agents/__init__.py` imports two modules that do not exist, and `utils/launcher.py` imports `serl_launcher.serl_launcher...`. Both are fixed by `apply_patches.py`.
- **Fixed in this setup:**
  - TensorFlow 2.15 needs `protobuf<4`, so `tensorflow-metadata` is pinned to 1.14.0 and `wandb` to 0.16.6.
  - ConRFT's agent asserts 7-D actions, so the simulator uses that layout.
  - ConRFT's replay buffer hard-codes 384-d Octo-small embeddings, so the tiny stand-in Octo uses 384 too.
  - jaxlib 0.4.20+cuda11 needs cuDNN 8.x, so the GPU install pins `nvidia-cudnn-cu11==8.9.6.50` (unpinned, uv picks cuDNN 9 and jax falls back to the CPU).
  - The `run_*.sh` scripts read `XLA_PYTHON_CLIENT_MEM_FRACTION` from the environment, keeping the old values as defaults.
  - `apply_patches.py` also changes `jnp.array` to `jnp.asarray` where `train_conrft_octo.py` moves the agent to the GPU, so it is not copied twice.

## GPU test (24 Oct 2026, RTX 3060 Laptop 6 GB, WSL2)
The GPU install, the real Octo-small weights, `test_sim_env.py` (5/5) and demo recording all work.
Pretraining runs out of GPU memory at every batch size tried (256, 64, 32): the agent's train state
includes the T5-base text encoder, which does not fit next to Octo-small and the critics in 6 GB.
Use a GPU with more memory for stages 1 and 2.
