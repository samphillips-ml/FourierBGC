"""Reusable seeding utility for reproducible training runs. Nothing in this
repo was seeded before this (RawCNNProbe, RawCNNScalarProbe, FourierBGC,
ppcon_no_scalar, etc. all ran unseeded, single-shot), so wiring this in is
purely additive: train.py's --seed defaults to None, which skips set_seed
and passes generator=None to the DataLoader, reproducing the exact old
(unseeded) behavior for every existing invocation. Opt in per run with
--seed N.
"""
import random

import numpy as np
import torch


def set_seed(seed):
    """Seeds python's random, numpy, and torch (CPU + all CUDA devices, if
    available). Does not touch MPS RNG state (torch has no
    manual_seed_all-equivalent for MPS as of this codebase's torch version);
    this repo's train.py forces device="cpu" regardless, so that gap
    doesn't affect the runs this is used for."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed):
    """A torch.Generator for DataLoader(..., generator=...), so a shuffled
    loader's batch order is reproducible for a given seed independent of
    global RNG state."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
