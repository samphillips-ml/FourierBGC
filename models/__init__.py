"""
The ablation spine: one module per model in the manuscript.

The manuscript's names and the code's historical names diverged. This registry
is the single mapping between them; nothing else should hard-code a model name.

    paper name            registry key            class
    --------------------------------------------------------------------
    PPCon                 (third_party/ppcon)     -- released checkpoint
    PPCon-NoCoord         ppcon_no_coord          -- loader, not a class
    CNN-NoCoord           cnn_no_coord            CNNNoCoord
    CNN-RawCoord          cnn_raw_coord           CNNRawCoord
    CNN-MLPCoord          cnn_mlp_coord           CNNMLPCoord
    FourierBGC-Broadcast  fourierbgc_broadcast    FourierBGCBroadcast
    FourierBGC            fourierbgc              FourierBGC

LEGACY_NAMES maps the flags the SLURM scripts and older checkpoints used onto
the registry keys, so `--model cnn_scalar` still resolves.
"""
from models.backbone import Conv1dMed, count_params
from models.cnn_no_coord import CNNNoCoord
from models.cnn_raw_coord import CNNRawCoord
from models.cnn_mlp_coord import CNNMLPCoord
from models.fourierbgc_broadcast import FourierBGCBroadcast
from models.fourierbgc import FourierBGC

REGISTRY = {
    "cnn_no_coord":         CNNNoCoord,
    "cnn_raw_coord":        CNNRawCoord,
    "cnn_mlp_coord":        CNNMLPCoord,
    "fourierbgc_broadcast": FourierBGCBroadcast,
    "fourierbgc":           FourierBGC,
}

# old flag -> registry key. Kept so existing commands and scripts keep working.
LEGACY_NAMES = {
    "cnn":                  "cnn_no_coord",
    "cnn_scalar":           "cnn_raw_coord",
    "cnn_mlpcoord":         "cnn_mlp_coord",
    "fourierbgc_with_year": "fourierbgc_broadcast",
    "fourierbgc_learned":   "fourierbgc",
    "ppcon_no_scalar":      "ppcon_no_coord",
}

# models needing a checkpoint directory and an epoch rather than a single file
CHECKPOINT_DIR_MODELS = ("ppcon", "ppcon_no_coord")


def resolve(name):
    """Canonical registry key for a paper name, registry key, or legacy flag."""
    return LEGACY_NAMES.get(name, name)


def make_model(name, **kwargs):
    key = resolve(name)
    if key == "cnn_mlp_coord_norm":
        return CNNMLPCoord(normalize=True, **kwargs)
    if key not in REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; known: {sorted(REGISTRY)} "
            f"(plus legacy aliases {sorted(LEGACY_NAMES)}, "
            f"and {CHECKPOINT_DIR_MODELS} which load from a directory)")
    return REGISTRY[key](**kwargs)


__all__ = ["Conv1dMed", "CNNNoCoord", "CNNRawCoord", "CNNMLPCoord",
           "FourierBGCBroadcast", "FourierBGC", "REGISTRY", "LEGACY_NAMES",
           "CHECKPOINT_DIR_MODELS", "resolve", "make_model", "count_params"]
