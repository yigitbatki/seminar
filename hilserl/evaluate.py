"""Evaluates a trained HIL-SERL checkpoint on sim_pick_cube without any expert help.

Runs the episodes once with sampled actions (as train_rlpd.py's own eval does) and once with
the policy's mode, on fixed evaluation seeds that do not depend on the training seed, and
writes the results as JSON.

usage (with the hil-serl venv active):
  python evaluate.py --ckpt_dir RUN/ckpt [--step N] [--episodes 100] [--out results.json]
The task settings (cube range, lift height, episode length, ...) come from the same
HILSERL_SIM_* environment variables as training.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MPLBACKEND"] = "Agg"
os.environ["HILSERL_SIM_HZ"] = "0"  # no real-time pacing during evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True, help="checkpoint folder (<run>/ckpt)")
parser.add_argument("--step", type=int, default=0, help="checkpoint step (default: latest)")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--modes", default="stochastic,deterministic")
parser.add_argument("--eval_seed", type=int, default=1000, help="cube positions; same for every run")
parser.add_argument("--out", default="results.json")
parser.add_argument("--workdir", default=os.environ.get("WORKDIR", os.path.expanduser("~/hilserl-work")))
args = parser.parse_args()
os.environ["HILSERL_SIM_SEED"] = str(args.eval_seed)

sys.path.insert(0, os.path.join(args.workdir, "hil-serl", "examples"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax.training import checkpoints  # noqa: E402

from experiments.mappings import CONFIG_MAPPING  # noqa: E402
from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm  # noqa: E402

config = CONFIG_MAPPING["sim_pick_cube"]()
env = config.get_environment(fake_env=True, save_video=False, classifier=False)  # no expert wrapper

# Same construction as train_rlpd.py, then the checkpoint is restored into agent.state.
agent = make_sac_pixel_agent_hybrid_single_arm(
    seed=args.eval_seed,
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

results = {"checkpoint_step": step, "eval_seed": args.eval_seed, "episodes": args.episodes,
           "cube_range": os.environ.get("HILSERL_SIM_CUBE_RANGE", "default")}
for mode in args.modes.split(","):
    env.unwrapped._cube_rng_state = np.random.RandomState(args.eval_seed).get_state()  # same cubes per mode
    rng = jax.random.PRNGKey(args.eval_seed)
    lengths, succeeded, t0 = [], [], time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, t = False, 0
        while not done:
            rng, key = jax.random.split(rng)
            action = np.asarray(jax.device_get(agent.sample_actions(
                observations=jax.device_put(obs), seed=key, argmax=(mode == "deterministic"))))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            t += 1
        lengths.append(t)
        succeeded.append(bool(info["succeed"]))
    k = sum(succeeded)
    results[mode] = {
        "successes": k,
        "success_rate": k / args.episodes,
        "mean_steps_to_success": float(np.mean([l for l, s in zip(lengths, succeeded) if s])) if k else None,
        "lengths": lengths,
        "succeeded": succeeded,
    }
    print(f"{mode}: {k}/{args.episodes} successes, mean steps to success "
          f"{results[mode]['mean_steps_to_success']}, {time.time() - t0:.0f}s", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print("saved", args.out, flush=True)
os._exit(0)  # skip MuJoCo's noisy EGL cleanup at exit
