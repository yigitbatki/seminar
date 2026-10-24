"""Builds demo sets from recorded demo files (lists of transitions from record_demos.py).

  python make_demo_set.py --out first5.pkl --first 5 demos20.pkl     # first 5 episodes
  python make_demo_set.py --out demos50.pkl demos20.pkl extra30.pkl  # concatenate files
Prints the number of episodes and transitions written.
"""
import argparse
import pickle as pkl


def episodes(transitions):
    """Splits a flat transition list at done=True."""
    eps, cur = [], []
    for tr in transitions:
        cur.append(tr)
        if tr["dones"]:
            eps.append(cur)
            cur = []
    return eps


parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--first", type=int, default=0, help="keep only the first N episodes (after concatenation)")
args = parser.parse_args()

eps = []
for path in args.inputs:
    with open(path, "rb") as f:
        eps += episodes(pkl.load(f))
if args.first:
    if args.first > len(eps):
        raise SystemExit(f"asked for {args.first} episodes but the inputs have only {len(eps)}")
    eps = eps[: args.first]
out = [tr for ep in eps for tr in ep]
with open(args.out, "wb") as f:
    pkl.dump(out, f)
print(f"wrote {len(eps)} episodes, {len(out)} transitions to {args.out}")
