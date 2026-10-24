"""Runs a list of HIL-SERL and ConRFT sim experiments one after another, resumable after a disconnect.

Each entry is a dict with RUN_NAME and the settings that differ from the method's defaults, e.g.
    {"RUN_NAME": "seed43", "SEED": 43}                               (HIL-SERL, run 2's settings)
    {"RUN_NAME": "wide_s42", "CUBE_RANGE": "wide"}
    {"RUN_NAME": "run2", "eval_only": True}
    {"RUN_NAME": "conrft_cal_s42", "METHOD": "conrft", "STAGE": "pretrain"}          (Cal-ConRFT)
    {"RUN_NAME": "conrft_sft_s42", "METHOD": "conrft", "STAGE": "pretrain", "Q_WEIGHT": 0}
    {"RUN_NAME": "conrft_hil_s42", "METHOD": "conrft", "INIT_FROM": "conrft_cal_s42"}  (HIL-ConRFT)
For every entry: skip it if <DRIVE_ROOT>/<RUN_NAME>/DONE exists; otherwise make sure its demo set
exists, train (HIL-SERL: run_training.sh; ConRFT: ../conrft-sim/run_conrft.sh, both resume a
partial run), evaluate the final checkpoint on fixed evaluation cubes, write results.json, then
DONE. summarize() builds the comparison table and learning curves from all finished runs.

Paths come from the environment: WORKDIR (hilserl setup.sh's folder), CONRFT_WORKDIR (conrft-sim
setup.sh's folder), DRIVE_ROOT (persistent results, e.g. MyDrive/hilserl-runs), LOCAL_ROOT (fast
scratch for running runs). CONRFT_TINY_OCTO=1 uses ConRFT's tiny stand-in Octo (plumbing tests).
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
CONRFT_HERE = os.path.join(os.path.dirname(HERE), "conrft-sim")

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
# ConRFT: same task settings; its own network (Octo-small), discount (fixed in its config) and losses.
# STAGE "pretrain" = offline on the demos (Cal-ConRFT: Q 0.1 / BC 1.0, SFT: Q 0 / BC 1.0);
# STAGE "online" = HIL-ConRFT from INIT_FROM's final checkpoint (Q 1.0 / BC 0.1;
# HG-DAgger: Q 0 / BC 1.0 with BC_DEMO_ONLY).
CONRFT_DEFAULTS = dict(
    DEFAULTS, DISCOUNT=0.98, ENCODER_TYPE="octo-small", STAGE="online", Q_WEIGHT=None, BC_WEIGHT=None,
    BC_DEMO_ONLY=False, PRETRAIN_STEPS=20000, INIT_FROM=None,
)
CONRFT_STAGE_WEIGHTS = {"pretrain": (0.1, 1.0), "online": (1.0, 0.1)}
DEMO_TASK_KEYS = ("CUBE_RANGE", "EXPERT_NOISE", "LIFT_HEIGHT", "MAX_EPISODE_STEPS")
RECORD_SEED = 1001  # demos recorded here never reuse run 2's demo episodes


def _paths():
    return os.environ["WORKDIR"], os.environ["DRIVE_ROOT"], os.environ.get("LOCAL_ROOT", "/content/runs")


def _conrft_workdir():
    return os.environ.get("CONRFT_WORKDIR", "/content/conrft-sim-work")


def _defaults(method):
    return CONRFT_DEFAULTS if method == "conrft" else DEFAULTS


def _settings(entry):
    method = entry.get("METHOD", "hilserl")
    if method not in ("hilserl", "conrft"):
        raise ValueError(f"{entry['RUN_NAME']}: METHOD must be hilserl or conrft")
    defaults = _defaults(method)
    unknown = set(entry) - set(defaults) - {"RUN_NAME", "eval_only", "METHOD"}
    if unknown:
        raise ValueError(f"{entry['RUN_NAME']}: unknown settings {sorted(unknown)}")
    settings = {**defaults, **{k: v for k, v in entry.items() if k in defaults}, "METHOD": method}
    if method == "conrft":
        if settings["STAGE"] not in CONRFT_STAGE_WEIGHTS:
            raise ValueError(f"{entry['RUN_NAME']}: STAGE must be pretrain or online")
        q, bc = CONRFT_STAGE_WEIGHTS[settings["STAGE"]]
        settings["Q_WEIGHT"] = q if settings["Q_WEIGHT"] is None else settings["Q_WEIGHT"]
        settings["BC_WEIGHT"] = bc if settings["BC_WEIGHT"] is None else settings["BC_WEIGHT"]
        if settings["STAGE"] == "online" and not settings["INIT_FROM"] and not entry.get("eval_only"):
            raise ValueError(f"{entry['RUN_NAME']}: an online ConRFT run needs INIT_FROM (a pretrain run)")
    return settings


def _env(settings, **extra):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({ENV_NAMES[k]: str(settings[k]) for k in ENV_NAMES})
    if settings.get("METHOD") == "conrft":
        env.update(STAGE=settings["STAGE"], Q_WEIGHT=str(settings["Q_WEIGHT"]), BC_WEIGHT=str(settings["BC_WEIGHT"]),
                   BC_DEMO_ONLY="1" if settings["BC_DEMO_ONLY"] else "0", PRETRAIN_STEPS=str(settings["PRETRAIN_STEPS"]))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _conrft_python():
    """The command that runs a ConRFT script from conrft/examples (with the tiny Octo if requested)."""
    tiny = os.environ.get("CONRFT_TINY_OCTO", "0") == "1"
    return f"python tiny_octo_smoke.py" if tiny else "python"


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


def _record(settings, n, seed, dest, cube_positions=None):
    """Records n expert demos with the given task settings and copies the file to dest."""
    if settings.get("METHOD") == "conrft":
        workdir = _conrft_workdir()
        ex = f"{workdir}/conrft/examples"
    else:
        workdir, _, _ = _paths()
        ex = f"{workdir}/hil-serl/examples"
    pattern = f"{ex}/demo_data/sim_pick_cube_{n}_demos_*.pkl"
    for old in glob.glob(pattern):
        os.remove(old)
    cmd = (f"source {workdir}/.venv/bin/activate && cd {ex} && "
           f"N_DEMOS={n} bash experiments/sim_pick_cube/run_record_demos.sh 2>&1 | grep -E 'saved|Error|Traceback'")
    extra = {"HILSERL_SIM_CUBE_POSITIONS": cube_positions} if cube_positions else {}
    if _run(cmd, _env({**settings, "SEED": seed}, **extra), shell=True) != 0 or not glob.glob(pattern):
        raise RuntimeError(f"recording {n} demos failed")
    shutil.copy(sorted(glob.glob(pattern))[-1], dest)


def _demo_set(cmd_args, dest):
    workdir, _, _ = _paths()
    tmp = dest + ".tmp"
    subprocess.run([f"{workdir}/.venv/bin/python", f"{HERE}/make_demo_set.py", "--out", tmp, *cmd_args], check=True)
    os.replace(tmp, dest)


def ensure_conrft_demos(settings):
    """ConRFT's demo files (256 px images, 2 stacked frames, Octo embeddings), built on first use.

    The base set is 20 demos at run 2's 20 cube positions (recovered from run 2's demo file), so
    both methods learn from the same cubes; 5/10 demos are its first episodes, 50 demos add 30
    new ones. Any other task setting gets its own recording.
    """
    _, drive, _ = _paths()
    n = int(settings["N_DEMOS"])
    sets = f"{drive}/demos/conrft"
    os.makedirs(sets, exist_ok=True)
    if all(settings[k] == DEFAULTS[k] for k in DEMO_TASK_KEYS):
        base = f"{sets}/n20_run2cubes.pkl"
        if not os.path.exists(base):
            run2 = sorted(glob.glob(f"{drive}/demos/sim_pick_cube_20_demos_*.pkl"))
            if not run2:
                raise RuntimeError(f"run 2's demos are missing from {drive}/demos")
            cubes = f"{drive}/demos/run2_cubes.npy"
            if not os.path.exists(cubes):
                subprocess.run([f"{_conrft_workdir()}/.venv/bin/python", f"{HERE}/cube_positions_from_demos.py",
                                run2[-1], cubes], check=True)
            _record(settings, 20, DEFAULTS["SEED"], base, cube_positions=cubes)
        if n == 20:
            return base
        dest = f"{sets}/n{n}.pkl"
        if not os.path.exists(dest):
            if n < 20:
                _demo_set(["--first", str(n), base], dest)
            else:
                extra = f"{sets}/extra{n - 20}_seed{RECORD_SEED}.pkl"
                if not os.path.exists(extra):
                    _record(settings, n - 20, RECORD_SEED, extra)
                _demo_set([base, extra], dest)
        return dest
    name = "_".join(f"{k.lower()}{settings[k]}" for k in DEMO_TASK_KEYS if settings[k] != DEFAULTS[k])
    dest = f"{sets}/{name}_n{n}.pkl"
    if not os.path.exists(dest):
        _record(settings, n, RECORD_SEED, dest)
    return dest


def ensure_demos(settings):
    """Returns the demo file for these settings, building it on first use.

    Run 2's 20 demos are the base: 5/10 demos are its first episodes, 50 demos are those 20 plus
    30 new ones. Any other task setting (e.g. the wide cube range) gets its own recording.
    """
    if settings.get("METHOD") == "conrft":
        return ensure_conrft_demos(settings)
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


def _latest_step(ckpt_dir):
    steps = [int(p.rsplit("_", 1)[1]) for p in glob.glob(f"{ckpt_dir}/checkpoint_*") if p.rsplit("_", 1)[1].isdigit()]
    return max(steps) if steps else None


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

    conrft = settings["METHOD"] == "conrft"
    if not entry.get("eval_only"):
        demo = ensure_demos(settings)
        print(f"[{name}] training with {demo}", flush=True)
        extra = dict(RUN_DIR=f"{local}/{name}", SYNC_DIR=sync, DEMO_PATH=demo)
        if conrft:
            extra["WORKDIR"] = _conrft_workdir()
            if settings["STAGE"] == "online":
                init_dir = f"{drive}/{settings['INIT_FROM']}"
                init_step = _latest_step(f"{init_dir}/ckpt")
                if not os.path.exists(f"{init_dir}/DONE") or init_step is None:
                    raise RuntimeError(f"[{name}] needs the finished pretrain run {settings['INIT_FROM']} first")
                extra["INIT_CKPT"] = f"{init_dir}/ckpt/checkpoint_{init_step}"
            code = _run(["bash", f"{CONRFT_HERE}/run_conrft.sh"], _env(settings, **extra))
        else:
            code = _run(["bash", f"{HERE}/run_training.sh"], _env(settings, **extra))
        if conrft and settings["STAGE"] == "pretrain":
            done_step = _latest_step(f"{sync}/ckpt")
            if code != 0 or done_step != settings["PRETRAIN_STEPS"]:
                raise RuntimeError(f"[{name}] pretraining stopped at checkpoint {done_step} (exit code {code}); "
                                   "fix the cause and re-run the queue cell to resume")
        else:
            steps = _env_steps(f"{sync}/logs/episodes.csv")
            if code != 0 or steps < settings["TARGET_ENV_STEPS"]:
                raise RuntimeError(f"[{name}] training stopped at {steps} env steps (exit code {code}); "
                                   "fix the cause and re-run the queue cell to resume")

    print(f"[{name}] evaluating {settings['EVAL_EPISODES']} episodes", flush=True)
    env = _env(settings, XLA_PYTHON_CLIENT_MEM_FRACTION=".3")
    if conrft:
        cwd = _conrft_workdir()
        cmd = (f"source {cwd}/.venv/bin/activate && cd {cwd}/conrft/examples && "
               f"{_conrft_python()} {CONRFT_HERE}/evaluate_conrft.py --workdir {cwd} --ckpt_dir {sync}/ckpt "
               f"--episodes {settings['EVAL_EPISODES']} --out {sync}/results.json")
        code = _run(cmd, env, shell=True)
    else:
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
    for i, entry in enumerate(queue):
        settings = _settings(entry)  # fail early on typos
        init = settings.get("INIT_FROM")
        if init and init not in names[:i] and not os.path.exists(f"{os.environ['DRIVE_ROOT']}/{init}/DONE"):
            raise ValueError(f"{entry['RUN_NAME']}: INIT_FROM {init} must come earlier in the queue or be done")
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
        method = settings.get("METHOD", res.get("method", "hilserl"))
        defaults = _defaults(method)
        if method == "conrft":
            label = settings.get("STAGE", "?")
            q, bc = settings.get("Q_WEIGHT"), settings.get("BC_WEIGHT")
            if label == "pretrain":
                label = "SFT" if q == 0 else "Cal-ConRFT"
            else:
                label = "HG-DAgger" if settings.get("BC_DEMO_ONLY") else "HIL-ConRFT"
            label += f" (q {q}, bc {bc}" + (f", from {settings['INIT_FROM']})" if settings.get("INIT_FROM") else ")")
        else:
            label = "HIL-SERL"
        skip = {"STAGE", "Q_WEIGHT", "BC_WEIGHT", "BC_DEMO_ONLY", "INIT_FROM", "METHOD"}
        changed = {k: v for k, v in settings.items() if k in defaults and k not in skip and v != defaults[k]}
        row = {"run": name, "method": label,
               "settings": ", ".join(f"{k}={v}" for k, v in changed.items()) or "defaults"}
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
