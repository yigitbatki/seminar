"""Fixes and small additions to the public ConRFT repository (idempotent).

1. serl_launcher/agents/__init__.py imports sac_hybrid_single / sac_hybrid_dual,
   which are not in the repository.
2. serl_launcher/utils/launcher.py imports `serl_launcher.serl_launcher...`,
   which only resolves when the repository root happens to be on sys.path.
3. examples/train_conrft_octo.py copies every agent leaf with jnp.array before
   device_put, so the agent briefly sits on the GPU twice; jnp.asarray avoids the copy.
4. --bc_demo_only (new flag, for the HG-DAgger baseline): the actor's BC loss uses only the
   demo/intervention half of each online batch instead of the whole batch.
5. Pretraining always saves a checkpoint at its last step (upstream only saves when the step
   is a multiple of checkpoint_period).
6. Update-to-data lock (same as ../hilserl/utd_lock.patch): the online learner publishes its
   step in HILSERL_SIM_UTD_FILE, the actor waits when it is too far ahead. The learner's steps
   continue from pretraining, so HILSERL_SIM_UTD_OFFSET (the pretrain step) is subtracted.

Usage: python apply_patches.py /path/to/conrft
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "serl_launcher" / "serl_launcher"

agents_init = root / "agents" / "__init__.py"
agents_init.write_text(
    "from .continuous.bc import BCAgent\n"
    "from .continuous.sac import SACAgent\n"
    "\n"
    "agents = {\n"
    '    "bc": BCAgent,\n'
    '    "sac": SACAgent,\n'
    "}\n"
)

launcher = root / "utils" / "launcher.py"
text = launcher.read_text()
text = text.replace(
    "from serl_launcher.serl_launcher.agents.continuous.conrft_single_octo_cp",
    "from serl_launcher.agents.continuous.conrft_single_octo_cp",
)
launcher.write_text(text)

def patch(path, old, new, marker):
    """Replaces old with new once; skips files that already contain marker."""
    text = path.read_text()
    if marker in text:
        return
    if old not in text:
        raise SystemExit(f"{path}: expected text not found, upstream code changed?")
    path.write_text(text.replace(old, new, 1))


train = pathlib.Path(sys.argv[1]) / "examples" / "train_conrft_octo.py"
patch(train, "jnp.array, agent), sharding.replicate())", "jnp.asarray, agent), sharding.replicate())",
      "jnp.asarray, agent), sharding.replicate())")
patch(train, '''flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")''',
      '''flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")
flags.DEFINE_boolean("bc_demo_only", False,
                     "BC loss only on the demo/intervention half of online batches (HG-DAgger).")''',
      "bc_demo_only")
patch(train, '''    # replicate agent across devices''',
      '''    if FLAGS.bc_demo_only:
        agent = agent.replace(config={**agent.config, "bc_demo_only": True})

    # replicate agent across devices''',
      "agent.replace(config={**agent.config, \"bc_demo_only\": True})")
patch(train, '''        print_green("Pretraining done")''',
      '''        if not checkpoints.latest_checkpoint(FLAGS.checkpoint_path) or not os.path.exists(
                os.path.join(FLAGS.checkpoint_path, f"checkpoint_{FLAGS.pretrain_steps}")):
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=FLAGS.pretrain_steps, keep=100)
        print_green("Pretraining done")''',
      "step=FLAGS.pretrain_steps, keep=100")
patch(train, r'''def print_green(x):
    return print("\033[92m {}\033[00m".format(x))
''', r'''def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


UTD_FILE = os.environ.get("HILSERL_SIM_UTD_FILE", "")
UTD = float(os.environ.get("HILSERL_SIM_UTD", "0"))
UTD_SLACK = int(os.environ.get("HILSERL_SIM_UTD_SLACK", "500"))
UTD_OFFSET = int(os.environ.get("HILSERL_SIM_UTD_OFFSET", "0"))


def write_learner_step(step):
    if UTD_FILE:
        with open(UTD_FILE + ".tmp", "w") as f:
            f.write(str(step))
        os.replace(UTD_FILE + ".tmp", UTD_FILE)


def read_learner_step():
    try:
        with open(UTD_FILE) as f:
            return int(f.read())
    except (OSError, ValueError):
        return 0
''', "HILSERL_SIM_UTD_FILE")
patch(train, '''    for step in pbar:
        timer.tick("total")
''', '''    for step in pbar:
        if UTD_FILE and UTD > 0:
            with timer.context("wait_for_learner"):
                while step > UTD_SLACK + read_learner_step() / UTD:
                    time.sleep(0.05)
        timer.tick("total")
''', "wait_for_learner")
patch(train, '''    online_start_step = start_step

''', '''    online_start_step = start_step
    write_learner_step(start_step - UTD_OFFSET)

''', "write_learner_step(start_step - UTD_OFFSET)")
patch(train, '''            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
''', '''            server.publish_network(agent.state.params)
        if step % 10 == 0:
            write_learner_step(step - UTD_OFFSET)

        if step % config.log_period == 0 and wandb_logger:
''', "write_learner_step(step - UTD_OFFSET)")

agent_file = root / "agents" / "continuous" / "conrft_single_octo_cp.py"
patch(agent_file, '''        recon_loss = (mean_flat(recon_diffs) * weights).mean()''',
      '''        if self.config.get("bc_demo_only", False):
            # Online batches are [online half, demo/intervention half]; imitate only the latter.
            demo_half = (jnp.arange(batch_size) >= batch_size // 2).astype(weights.dtype)
            recon_loss = (mean_flat(recon_diffs) * weights * demo_half).sum() / demo_half.sum()
        else:
            recon_loss = (mean_flat(recon_diffs) * weights).mean()''',
      "bc_demo_only")
print("patched", agents_init, launcher, train, "and", agent_file)
