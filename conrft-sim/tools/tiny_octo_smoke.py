"""Offline smoke test: run a ConRFT script with a tiny, randomly initialised Octo.

This only checks that the plumbing works (shapes, demo format, agent updates,
learner/actor loop) when the real Octo weights cannot be downloaded. It does not
learn anything useful. Run from conrft/examples, for example:

    python ../../conrft-sim/tools/tiny_octo_smoke.py record_demos_octo.py --exp_name=sim_pick_cube --successes_needed=2
"""
import os
import runpy
import sys

import jax
import numpy as np

from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import SmallStem16
from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec


def make_tiny_octo():
    _, transformer_kwargs = common_transformer_sizes("dummy")
    token_embedding_size = 384  # ConRFT's replay buffer assumes Octo-small's 384-d embeddings
    config = {
        "model": dict(
            observation_tokenizers={
                "primary": ModuleSpec.create(
                    ImageTokenizer, obs_stack_keys=["image_primary"], task_stack_keys=[],
                    encoder=ModuleSpec.create(SmallStem16),
                ),
                "wrist": ModuleSpec.create(
                    ImageTokenizer, obs_stack_keys=["image_wrist"], task_stack_keys=[],
                    encoder=ModuleSpec.create(SmallStem16),
                ),
            },
            task_tokenizers={},
            readouts={"action": 1},
            heads={
                "action": ModuleSpec.create(
                    DiffusionActionHead, readout_key="readout_action", use_map=False,
                    action_horizon=4, action_dim=7, n_diffusion_samples=1, dropout_rate=0.0,
                )
            },
            token_embedding_size=token_embedding_size,
            transformer_kwargs=transformer_kwargs,
            max_horizon=10,
            repeat_task_tokens=True,
            use_correct_attention=True,
        ),
        "text_processor": None,
    }
    example_batch = {
        "observation": {
            "image_primary": np.zeros((1, 2, 256, 256, 3), np.uint8),
            "image_wrist": np.zeros((1, 2, 128, 128, 3), np.uint8),
            "timestep_pad_mask": np.ones((1, 2), bool),
            "pad_mask_dict": {
                "image_primary": np.ones((1, 2), bool),
                "image_wrist": np.ones((1, 2), bool),
            },
        },
        "task": {"pad_mask_dict": {}},
        "action": np.zeros((1, 2, 4, 7), np.float32),
    }
    return OctoModel.from_config(config, example_batch, rng=jax.random.PRNGKey(0))


_TINY = make_tiny_octo()
OctoModel.load_pretrained = classmethod(lambda cls, *a, **k: _TINY)
OctoModel.create_tasks = lambda self, goals=None, texts=None: {"pad_mask_dict": {}}

script, *args = sys.argv[1:]
sys.argv = [script, *args]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
