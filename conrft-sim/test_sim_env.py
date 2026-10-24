"""Smoke test for the simulated ConRFT task (no Octo needed).

Checks observation shapes, then runs the scripted expert in demo mode and
reports its success rate. Run from conrft/examples:
    MUJOCO_GL=egl python test_sim_env.py --episodes 10
"""
import argparse
import os
import time

import numpy as np

os.environ.setdefault("HILSERL_SIM_MODE", "demo")
os.environ.setdefault("HILSERL_SIM_HZ", "0")
from experiments.mappings import CONFIG_MAPPING  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save_frames", default="", help="optional .npz path for one episode's images")
    args = parser.parse_args()

    config = CONFIG_MAPPING["sim_pick_cube"]()
    env = config.get_environment(fake_env=False, classifier=True, stack_obs_num=2)
    obs, _ = env.reset()
    print("observation space:", {k: v.shape for k, v in env.observation_space.spaces.items()})
    print("observation:", {k: (v.shape, v.dtype) for k, v in obs.items()})
    print("action space:", env.action_space.shape)

    successes, lengths, interventions = 0, [], 0
    frames = []
    t0 = time.time()
    steps = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, t = False, 0
        while not done:
            obs, rew, done, trunc, info = env.step(np.zeros(env.action_space.shape))
            interventions += "intervene_action" in info
            t += 1
            steps += 1
            if ep == 0 and args.save_frames:
                frames.append(obs["side_policy_256"][-1])
        successes += int(info["succeed"])
        lengths.append(t)
        print(f"episode {ep}: succeed={info['succeed']} length={t} reward={rew}")
    dt = time.time() - t0
    print(f"expert success rate: {successes}/{args.episodes}, mean length {np.mean(lengths):.1f}, "
          f"intervened steps {interventions}/{steps}, {steps / dt:.1f} env steps/s")
    if frames:
        np.savez_compressed(args.save_frames, frames=np.stack(frames))
        print("saved frames to", args.save_frames)


if __name__ == "__main__":
    main()
