"""Fixes two import bugs and one memory issue in the public ConRFT repository (idempotent).

1. serl_launcher/agents/__init__.py imports sac_hybrid_single / sac_hybrid_dual,
   which are not in the repository.
2. serl_launcher/utils/launcher.py imports `serl_launcher.serl_launcher...`,
   which only resolves when the repository root happens to be on sys.path.
3. examples/train_conrft_octo.py copies every agent leaf with jnp.array before
   device_put, so the agent briefly sits on the GPU twice; jnp.asarray avoids the copy.

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
train = pathlib.Path(sys.argv[1]) / "examples" / "train_conrft_octo.py"
text = train.read_text()
text = text.replace(
    "jnp.array, agent), sharding.replicate())",
    "jnp.asarray, agent), sharding.replicate())",
)
train.write_text(text)
print("patched", agents_init, launcher, "and", train)
