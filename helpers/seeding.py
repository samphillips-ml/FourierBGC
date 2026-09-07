"""Seeding for reproducible runs. Every model in the paper was trained with
seeds 0-4; without --seed, training is unseeded."""
import random

import numpy as np
import torch


def set_seed(seed):
    """Seeds python's random, numpy and torch, including CUDA. MPS is not
    seeded; it is never used here."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_generator(seed):
    """A torch.Generator for DataLoader(..., generator=...), so shuffled batch
    order is reproducible independent of global RNG state."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
