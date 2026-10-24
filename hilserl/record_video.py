"""Records an MP4 of the trained HIL-SERL policy (or the scripted expert) on sim_pick_cube.

The policy still sees its usual 128x128 cameras; the video shows a 640x480 render of the
front camera next to the policy's wrist view, with episode, step and success overlaid.

usage (with the hil-serl venv active):
  python record_video.py --ckpt_dir RUN/ckpt [--step N] [--episodes 5] [--out video.mp4]
  python record_video.py --expert --episodes 1 --out expert.mp4     # scripted expert instead
Add --stochastic to sample actions instead of taking the policy's mode.
"""
import argparse
import glob
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MPLBACKEND"] = "Agg"
os.environ["HILSERL_SIM_HZ"] = "0"            # no real-time pacing while recording
os.environ["HILSERL_SIM_INTERVENE_PROB"] = "0"  # no expert takeovers in policy episodes
os.environ.pop("HILSERL_SIM_EPISODE_LOG", None)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", help="checkpoint folder (<run>/ckpt)")
parser.add_argument("--step", type=int, default=0, help="checkpoint step (default: latest)")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--out", default="video.mp4")
parser.add_argument("--expert", action="store_true", help="record the scripted expert instead of the policy")
parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mode")
parser.add_argument("--fps", type=int, default=10, help="10 = real time (the env runs at 10 Hz)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--workdir", default=os.environ.get("WORKDIR", os.path.expanduser("~/hilserl-work")))
args = parser.parse_args()
if os.environ.get("HILSERL_SIM_MODE") is None or args.expert:
    os.environ["HILSERL_SIM_MODE"] = "demo" if args.expert else "online"

sys.path.insert(0, os.path.join(args.workdir, "hil-serl", "examples"))
import cv2  # noqa: E402
import imageio  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from flax.training import checkpoints  # noqa: E402

from experiments.mappings import CONFIG_MAPPING  # noqa: E402

config = CONFIG_MAPPING["sim_pick_cube"]()
# The expert wrapper is only added for real (non-fake) envs; the policy runs on the bare env.
env = config.get_environment(fake_env=not args.expert, save_video=False, classifier=False)
sim = env.unwrapped._env  # franka_sim PandaPickCubeGymEnv

agent = None
if not args.expert:
    from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm

    # Same construction as train_rlpd.py, then the checkpoint is restored into agent.state.
    agent = make_sac_pixel_agent_hybrid_single_arm(
        seed=args.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
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
    agent = agent.replace(state=checkpoints.restore_checkpoint(ckpt_dir, agent.state, step=step))
    print(f"loaded {ckpt_dir}/checkpoint_{step}", flush=True)

# A larger render of the front camera for the video (the policy keeps its 128 px input).
H, W = 480, 640
sim._model.vis.global_.offwidth = max(sim._model.vis.global_.offwidth, W)
sim._model.vis.global_.offheight = max(sim._model.vis.global_.offheight, H)
renderer = mujoco.Renderer(sim._model, height=H, width=W)
front_cam = sim.camera_id[0]


def frame(obs, ep, t, status):
    renderer.update_scene(sim._data, camera=front_cam)
    big = renderer.render()
    wrist = cv2.resize(np.asarray(obs["wrist"])[-1], (H, H), interpolation=cv2.INTER_NEAREST)
    img = np.concatenate([big, wrist], axis=1).copy()
    who = "scripted expert" if args.expert else ("policy (stochastic)" if args.stochastic else "policy")
    lines = [f"{who}  episode {ep + 1}/{args.episodes}  step {t}", status]
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
rng = jax.random.PRNGKey(args.seed)
successes = 0
for ep in range(args.episodes):
    obs, _ = env.reset()
    done, t, status = False, 0, ""
    writer.append_data(frame(obs, ep, t, status))
    while not done:
        if agent is None:
            action = np.zeros(env.action_space.shape)  # the expert wrapper replaces it
        else:
            rng, key = jax.random.split(rng)
            action = np.asarray(jax.device_get(agent.sample_actions(
                observations=jax.device_put(obs), seed=key, argmax=not args.stochastic)))
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
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
