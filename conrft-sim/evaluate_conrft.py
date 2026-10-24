"""Evaluates a trained ConRFT checkpoint on sim_pick_cube without any expert help.

Same protocol and results.json schema as hilserl/evaluate.py: fixed evaluation cubes (seed 1000,
independent of the training seed), success rate and steps to success. ConRFT's consistency
policy always samples (its sample_actions has no mode/argmax option), so only the "stochastic"
entry is filled and "deterministic" is null.

usage (with the conrft venv active, from conrft/examples; prefix with `python tiny_octo_smoke.py`
instead of `python` to use the tiny stand-in Octo):
  python evaluate_conrft.py --ckpt_dir RUN/ckpt [--step N] [--episodes 100] [--out results.json]
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
os.environ.pop("HILSERL_SIM_CUBE_POSITIONS", None)  # evaluation always uses its own cubes

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True, help="checkpoint folder (<run>/ckpt)")
parser.add_argument("--step", type=int, default=0, help="checkpoint step (default: latest)")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--eval_seed", type=int, default=1000, help="cube positions; same for every run")
parser.add_argument("--out", default="results.json")
parser.add_argument("--workdir", default=os.environ.get("WORKDIR", os.path.expanduser("~/conrft-sim-work")))
args = parser.parse_args()
os.environ["HILSERL_SIM_SEED"] = str(args.eval_seed)

sys.path.insert(0, os.path.join(args.workdir, "conrft", "examples"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
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
ckpt = checkpoints.restore_checkpoint(ckpt_dir, agent.state, step=step)
agent = agent.replace(state=agent.state.replace(params=ckpt.params, target_params=ckpt.target_params))
print(f"loaded {ckpt_dir}/checkpoint_{step}", flush=True)

env.unwrapped._cube_rng_state = np.random.RandomState(args.eval_seed).get_state()
rng = jax.random.PRNGKey(args.eval_seed)
lengths, succeeded, t0 = [], [], time.time()
for ep in range(args.episodes):
    obs, _ = env.reset()
    done, t = False, 0
    while not done:
        rng, key = jax.random.split(rng)
        action, _ = agent.sample_actions(observations=jax.device_put(obs), tasks=jax.device_put(tasks), seed=key)
        obs, reward, terminated, truncated, info = env.step(np.asarray(jax.device_get(action)))
        done = terminated or truncated
        t += 1
    lengths.append(t)
    succeeded.append(bool(info["succeed"]))
k = sum(succeeded)
results = {
    "method": "conrft", "checkpoint_step": step, "eval_seed": args.eval_seed, "episodes": args.episodes,
    "cube_range": os.environ.get("HILSERL_SIM_CUBE_RANGE", "default"),
    "stochastic": {
        "successes": k,
        "success_rate": k / args.episodes,
        "mean_steps_to_success": float(np.mean([l for l, s in zip(lengths, succeeded) if s])) if k else None,
        "lengths": lengths,
        "succeeded": succeeded,
    },
    "deterministic": None,
    "note": "ConRFT's consistency policy always samples; there is no deterministic mode",
}
print(f"stochastic: {k}/{args.episodes} successes, mean steps to success "
      f"{results['stochastic']['mean_steps_to_success']}, {time.time() - t0:.0f}s", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=1)
print("saved", args.out, flush=True)
os._exit(0)  # skip MuJoCo's noisy EGL cleanup at exit
