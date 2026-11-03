"""Records an MP4 of a trained ConRFT policy on the shared sim task (cube or peg).

The ConRFT counterpart of ../hilserl/record_video.py: a 640x480 render of the front camera next
to the 128 px wrist camera the policy sees, with episode, step and success overlaid. The policy
always samples (ConRFT has no deterministic mode). Episodes start from the evaluation cubes or
fixtures (seed 1000), so they match evaluate_conrft.py's first episodes.

usage (with the conrft venv active, from conrft/examples; HILSERL_SIM_TASK etc. as for the run):
  python record_video_conrft.py --ckpt_dir RUN/ckpt [--step N] [--episodes 5] [--out video.mp4]
"""
import argparse
import glob
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MPLBACKEND"] = "Agg"
os.environ["HILSERL_SIM_HZ"] = "0"  # no real-time pacing while recording
os.environ.pop("HILSERL_SIM_CUBE_POSITIONS", None)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True, help="checkpoint folder (<run>/ckpt)")
parser.add_argument("--step", type=int, default=0, help="checkpoint step (default: latest)")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--eval_seed", type=int, default=1000, help="cube or fixture positions")
parser.add_argument("--fps", type=int, default=10, help="10 = real time (the env runs at 10 Hz)")
parser.add_argument("--label", default="ConRFT policy", help="shown in the overlay")
parser.add_argument("--out", default="video.mp4")
parser.add_argument("--workdir", default=os.environ.get("WORKDIR", os.path.expanduser("~/conrft-sim-work")))
args = parser.parse_args()
os.environ["HILSERL_SIM_SEED"] = str(args.eval_seed)

sys.path.insert(0, os.path.join(args.workdir, "conrft", "examples"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import cv2  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from flax.serialization import from_state_dict  # noqa: E402
from flax.training import checkpoints  # noqa: E402
from octo.model.octo_model import OctoModel  # noqa: E402

from experiments.mappings import CONFIG_MAPPING  # noqa: E402
from serl_launcher.utils.launcher import make_conrft_octo_cp_pixel_agent_single_arm  # noqa: E402

config = CONFIG_MAPPING["sim_pick_cube"]()
env = config.get_environment(fake_env=True, save_video=False, classifier=False, stack_obs_num=2)  # no expert

# Same construction as train_conrft_octo.py, then the checkpoint's parameters are restored.
octo_model = OctoModel.load_pretrained(config.octo_path)
tasks = octo_model.create_tasks(texts=[config.task_desc])
agent = make_conrft_octo_cp_pixel_agent_single_arm(
    seed=args.eval_seed,
    sample_obs=env.observation_space.sample(),
    sample_action=env.action_space.sample(),
    sample_tasks=tasks,
    octo_model=octo_model,
    image_keys=config.image_keys,
    encoder_type=config.encoder_type,
    discount=config.discount,
)
agent = jax.device_put(jax.tree_map(jnp.asarray, agent))
ckpt_dir = os.path.abspath(args.ckpt_dir)
steps = [int(re.sub(r".*_", "", p)) for p in glob.glob(f"{ckpt_dir}/checkpoint_*") if re.search(r"checkpoint_\d+$", p)]
if not steps:
    sys.exit(f"no checkpoint_<step> folders in {ckpt_dir}")
step = args.step or max(steps)
ckpt = checkpoints.restore_checkpoint(ckpt_dir, None, step=step)  # no target: older checkpoints have more opt_states
agent = agent.replace(state=agent.state.replace(params=from_state_dict(agent.state.params, ckpt["params"]),
                                                target_params=from_state_dict(agent.state.target_params, ckpt["target_params"])))
print(f"loaded {ckpt_dir}/checkpoint_{step}", flush=True)

sim = env.unwrapped._env  # franka_sim PandaPickCubeGymEnv
# A larger render of the front camera for the video (the policy keeps its own camera images).
H, W = 480, 640
sim._model.vis.global_.offwidth = max(sim._model.vis.global_.offwidth, W)
sim._model.vis.global_.offheight = max(sim._model.vis.global_.offheight, H)
renderer = mujoco.Renderer(sim._model, height=H, width=W)
front_cam = sim.camera_id[0]
wrist_key = config.image_keys[1]


def frame(obs, ep, t, status):
    renderer.update_scene(sim._data, camera=front_cam)
    big = renderer.render()
    wrist = cv2.resize(np.asarray(obs[wrist_key])[-1], (H, H), interpolation=cv2.INTER_NEAREST)
    img = np.concatenate([big, wrist], axis=1).copy()
    lines = [f"{args.label}  episode {ep + 1}/{args.episodes}  step {t}", status]
    for i, text in enumerate(lines):
        y = 28 + 30 * i
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        color = (60, 220, 60) if status.startswith("success") else (255, 255, 255)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color if i else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "wrist camera (policy input, 128 px)", (W + 12, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
env.unwrapped._cube_rng_state = np.random.RandomState(args.eval_seed).get_state()
rng = jax.random.PRNGKey(args.eval_seed)
successes = 0
for ep in range(args.episodes):
    obs, _ = env.reset()
    done, t, status = False, 0, ""
    writer.append_data(frame(obs, ep, t, status))
    while not done:
        rng, key = jax.random.split(rng)
        action, _ = agent.sample_actions(observations=jax.device_put(obs), tasks=jax.device_put(tasks), seed=key)
        obs, reward, terminated, truncated, info = env.step(np.asarray(jax.device_get(action)))
        done = terminated or truncated
        t += 1
        status = ("success" if info["succeed"] else "timeout") + f" after {t} steps" if done else ""
        writer.append_data(frame(obs, ep, t, status))
    for _ in range(args.fps):  # hold the last frame for a second
        writer.append_data(frame(obs, ep, t, status))
    successes += int(info["succeed"])
    print(f"episode {ep + 1}: {'success' if info['succeed'] else 'timeout'} in {t} steps", flush=True)
writer.close()
print(f"{successes}/{args.episodes} successes, saved {args.out}", flush=True)
os._exit(0)  # skip MuJoCo's noisy EGL cleanup at exit
