"""Runs a list of HIL-SERL sim experiments one after another, resumable after a disconnect.

Each entry is a dict with RUN_NAME and the settings that differ from run 2, e.g.
    {"RUN_NAME": "seed43", "SEED": 43}
    {"RUN_NAME": "wide_s42", "CUBE_RANGE": "wide"}
    {"RUN_NAME": "run2", "eval_only": True}
For every entry: skip it if <DRIVE_ROOT>/<RUN_NAME>/DONE exists; otherwise make sure its demo set
exists, train with run_training.sh (which resumes a partial run), evaluate the final checkpoint
(evaluate.py: stochastic and deterministic, fixed eval seeds), write results.json, then DONE.
summarize() builds the comparison table and learning curves from all finished runs.

Paths come from the environment: WORKDIR (setup.sh's folder), DRIVE_ROOT (persistent results,
e.g. MyDrive/hilserl-runs), LOCAL_ROOT (fast scratch for running runs).
"""
import glob
import json
import math
import os
import shutil
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Run 2's settings; every entry starts from these.
DEFAULTS = dict(
    SEED=42, CUBE_RANGE="default", N_DEMOS=20, INTERVENE_PROB=0.2, TAKEOVER_STEPS=0, EXPERT_NOISE=0.1,
    LIFT_HEIGHT=0.1, MAX_EPISODE_STEPS=100, DISCOUNT=0.97, CTA_RATIO=2, ENCODER_TYPE="resnet-pretrained",
    HZ=10, TARGET_ENV_STEPS=30000, EVAL_EPISODES=100,
)
ENV_NAMES = dict(
    SEED="HILSERL_SIM_SEED", CUBE_RANGE="HILSERL_SIM_CUBE_RANGE", INTERVENE_PROB="HILSERL_SIM_INTERVENE_PROB",
    TAKEOVER_STEPS="HILSERL_SIM_TAKEOVER_STEPS", EXPERT_NOISE="HILSERL_SIM_EXPERT_NOISE",
    LIFT_HEIGHT="HILSERL_SIM_LIFT_HEIGHT", MAX_EPISODE_STEPS="HILSERL_SIM_MAX_EPISODE_STEPS",
    DISCOUNT="HILSERL_SIM_DISCOUNT", CTA_RATIO="HILSERL_SIM_CTA_RATIO", ENCODER_TYPE="HILSERL_SIM_ENCODER",
    HZ="HILSERL_SIM_HZ", TARGET_ENV_STEPS="TARGET_ENV_STEPS",
)
DEMO_TASK_KEYS = ("CUBE_RANGE", "EXPERT_NOISE", "LIFT_HEIGHT", "MAX_EPISODE_STEPS")
RECORD_SEED = 1001  # demos recorded here never reuse run 2's demo episodes


def _paths():
    return os.environ["WORKDIR"], os.environ["DRIVE_ROOT"], os.environ.get("LOCAL_ROOT", "/content/runs")


def _settings(entry):
    unknown = set(entry) - set(DEFAULTS) - {"RUN_NAME", "eval_only"}
    if unknown:
        raise ValueError(f"{entry['RUN_NAME']}: unknown settings {sorted(unknown)}")
    return {**DEFAULTS, **{k: v for k, v in entry.items() if k in DEFAULTS}}


def _env(settings, **extra):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({ENV_NAMES[k]: str(settings[k]) for k in ENV_NAMES})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _run(cmd, env, shell=False):
    """Runs a command, streams its output, and forwards Ctrl-C/interrupt to it."""
    p = subprocess.Popen(cmd, env=env, shell=shell, executable="/bin/bash" if shell else None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        p.send_signal(signal.SIGINT)
        p.wait()
        raise
    return p.wait()


def _record(settings, n, seed, dest):
    """Records n expert demos with the given task settings and copies the file to dest."""
    workdir, _, _ = _paths()
    ex = f"{workdir}/hil-serl/examples"
    pattern = f"{ex}/demo_data/sim_pick_cube_{n}_demos_*.pkl"
    for old in glob.glob(pattern):
        os.remove(old)
    cmd = (f"source {workdir}/.venv/bin/activate && cd {ex} && "
           f"N_DEMOS={n} bash experiments/sim_pick_cube/run_record_demos.sh 2>&1 | grep -E 'saved|Error|Traceback'")
    if _run(cmd, _env({**settings, "SEED": seed}), shell=True) != 0 or not glob.glob(pattern):
        raise RuntimeError(f"recording {n} demos failed")
    shutil.copy(sorted(glob.glob(pattern))[-1], dest)


def _demo_set(cmd_args, dest):
    workdir, _, _ = _paths()
    tmp = dest + ".tmp"
    subprocess.run([f"{workdir}/.venv/bin/python", f"{HERE}/make_demo_set.py", "--out", tmp, *cmd_args], check=True)
    os.replace(tmp, dest)


def ensure_demos(settings):
    """Returns the demo file for these settings, building it on first use.

    Run 2's 20 demos are the base: 5/10 demos are its first episodes, 50 demos are those 20 plus
    30 new ones. Any other task setting (e.g. the wide cube range) gets its own recording.
    """
    _, drive, _ = _paths()
    n = int(settings["N_DEMOS"])
    base = sorted(glob.glob(f"{drive}/demos/sim_pick_cube_20_demos_*.pkl"))
    sets = f"{drive}/demos/sets"
    os.makedirs(sets, exist_ok=True)
    default_task = all(settings[k] == DEFAULTS[k] for k in DEMO_TASK_KEYS)
    if default_task:
        if not base:
            raise RuntimeError(f"run 2's demos are missing from {drive}/demos (run cell 4 once with the defaults)")
        if n == 20:
            return base[-1]
        dest = f"{sets}/n{n}.pkl"
        if not os.path.exists(dest):
            if n < 20:
                _demo_set(["--first", str(n), base[-1]], dest)
            else:
                extra = f"{sets}/extra{n - 20}_seed{RECORD_SEED}.pkl"
                if not os.path.exists(extra):
                    _record(settings, n - 20, RECORD_SEED, extra)
                _demo_set([base[-1], extra], dest)
        return dest
    name = "_".join(f"{k.lower()}{settings[k]}" for k in DEMO_TASK_KEYS if settings[k] != DEFAULTS[k])
    dest = f"{sets}/{name}_n{n}.pkl"
    if not os.path.exists(dest):
        _record(settings, n, RECORD_SEED, dest)
    return dest


def _env_steps(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path) as f:
        rows = [line.rstrip("\r\n").split(",") for line in f][1:]
    return sum(int(r[4]) for r in rows if len(r) >= 6)


def run_entry(entry):
    workdir, drive, local = _paths()
    name = entry["RUN_NAME"]
    settings = _settings(entry)
    sync = f"{drive}/{name}"
    if os.path.exists(f"{sync}/DONE"):
        print(f"[{name}] already done, skipping", flush=True)
        return
    os.makedirs(sync, exist_ok=True)
    with open(f"{sync}/settings.json", "w") as f:
        json.dump({"RUN_NAME": name, "eval_only": bool(entry.get("eval_only")), **settings}, f, indent=1)

    if not entry.get("eval_only"):
        demo = ensure_demos(settings)
        print(f"[{name}] training with {demo}", flush=True)
        env = _env(settings, RUN_DIR=f"{local}/{name}", SYNC_DIR=sync, DEMO_PATH=demo)
        code = _run(["bash", f"{HERE}/run_training.sh"], env)
        steps = _env_steps(f"{sync}/logs/episodes.csv")
        if code != 0 or steps < settings["TARGET_ENV_STEPS"]:
            raise RuntimeError(f"[{name}] training stopped at {steps} env steps (exit code {code}); "
                               "fix the cause and re-run the queue cell to resume")

    print(f"[{name}] evaluating {settings['EVAL_EPISODES']} episodes x 2 modes", flush=True)
    env = _env(settings, XLA_PYTHON_CLIENT_MEM_FRACTION=".3")
    code = _run([f"{workdir}/.venv/bin/python", f"{HERE}/evaluate.py", "--ckpt_dir", f"{sync}/ckpt",
                 "--episodes", str(settings["EVAL_EPISODES"]), "--out", f"{sync}/results.json"], env)
    if code != 0:
        raise RuntimeError(f"[{name}] evaluation failed (exit code {code})")

    if not entry.get("eval_only"):
        # Keep the final checkpoint and logs; drop resume-only data to save Drive space.
        shutil.rmtree(f"{sync}/ckpt/buffer", ignore_errors=True)
        shutil.rmtree(f"{sync}/ckpt/demo_buffer", ignore_errors=True)
        ckpts = sorted(glob.glob(f"{sync}/ckpt/checkpoint_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
        for old in ckpts[:-1]:
            shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(f"{local}/{name}", ignore_errors=True)
    open(f"{sync}/DONE", "w").close()
    print(f"[{name}] done", flush=True)


def run_queue(queue):
    names = [e["RUN_NAME"] for e in queue]
    if len(set(names)) != len(names):
        raise ValueError("RUN_NAMEs must be unique")
    for entry in queue:
        _settings(entry)  # fail early on typos
    for i, entry in enumerate(queue):
        print(f"\n=== {i + 1}/{len(queue)}: {entry['RUN_NAME']} ===", flush=True)
        run_entry(entry)
    print("\nqueue finished", flush=True)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def summarize(names=None, out_prefix=None):
    """Prints and saves a table of all finished runs and plots their learning curves."""
    import matplotlib.pyplot as plt
    import pandas as pd

    drive = os.environ["DRIVE_ROOT"]
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{drive}/*/results.json"))
    if names:
        runs = [n for n in names if n in runs]
    rows, curves = [], {}
    for name in runs:
        res = json.load(open(f"{drive}/{name}/results.json"))
        settings_path = f"{drive}/{name}/settings.json"
        settings = json.load(open(settings_path)) if os.path.exists(settings_path) else {}
        changed = {k: v for k, v in settings.items() if k in DEFAULTS and v != DEFAULTS[k]}
        row = {"run": name, "settings": ", ".join(f"{k}={v}" for k, v in changed.items()) or "run 2 defaults"}
        for mode in ("stochastic", "deterministic"):
            m = res.get(mode)
            if not m:
                continue
            lo, hi = wilson(m["successes"], res["episodes"])
            row[f"{mode} success"] = f"{m['success_rate']:.2f} [{lo:.2f}, {hi:.2f}]"
            row[f"{mode} steps"] = round(m["mean_steps_to_success"], 1) if m["mean_steps_to_success"] else None
        csv = f"{drive}/{name}/logs/episodes.csv"
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df["env_steps"] = df["length"].cumsum()
            row["train env steps"] = int(df["length"].sum())
            row["intervention rate"] = round(float((df["intervened_steps"] > 0).mean()), 2)
            own = df[df["intervened_steps"] == 0]
            curves[name] = (own["env_steps"], own["succeed"].rolling(20, min_periods=1).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    out_prefix = out_prefix or f"{drive}/summary"
    table.to_csv(f"{out_prefix}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, (x, y) in curves.items():
        ax.plot(x, y, label=name, lw=1.2)
    ax.set_xlabel("env steps")
    ax.set_ylabel("policy-alone success (rolling 20 episodes)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=150)
    print(f"saved {out_prefix}.csv and {out_prefix}.png")
    return table, fig


if __name__ == "__main__":
    # python experiment_queue.py queue.json   (a JSON list of entries)
    run_queue(json.load(open(sys.argv[1])))
