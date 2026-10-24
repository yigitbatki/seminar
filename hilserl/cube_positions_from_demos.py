"""Recovers the cube x, y of every episode in a demo file (from record_demos.py).

The scripted expert closes the gripper only once the gripper is within 1 cm of the cube, so
the gripper position at that step is the cube position to within about a centimetre. Run 2's
demos were recorded before cube positions were seeded; this lets other demo sets (e.g. ConRFT's)
use the same cubes via HILSERL_SIM_CUBE_POSITIONS.

  python cube_positions_from_demos.py demos.pkl out.npy [--state_slice 1:4]
"""
import argparse
import pickle as pkl

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("demos")
parser.add_argument("out")
parser.add_argument("--state_slice", default="1:4",
                    help="gripper xyz inside the flattened state (gripper_pose, tcp_pose, tcp_vel)")
args = parser.parse_args()
lo, hi = map(int, args.state_slice.split(":"))

with open(args.demos, "rb") as f:
    transitions = pkl.load(f)
positions, closed = [], False
for tr in transitions:
    if not closed and np.asarray(tr["actions"])[-1] > 0.5:
        state = np.asarray(tr["observations"]["state"])[-1]  # newest frame
        positions.append(state[lo:hi][:2])
        closed = True
    if tr["dones"]:
        if not closed:
            raise SystemExit("an episode ended without the expert closing the gripper")
        closed = False
positions = np.asarray(positions)
np.save(args.out, positions)
print(f"{len(positions)} cube positions -> {args.out}")
print(np.round(positions, 3))
