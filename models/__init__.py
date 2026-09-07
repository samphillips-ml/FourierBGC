"""
Paper-to-code name registry for the ablation spine. See README.md for the
name mapping. LEGACY_NAMES resolves older flags (e.g. cnn_scalar) so existing
commands and scripts keep working.
"""
from models.backbone import Conv1dMed
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

LEGACY_NAMES = {
    "cnn":                  "cnn_no_coord",
    "cnn_scalar":           "cnn_raw_coord",
    "cnn_mlpcoord":         "cnn_mlp_coord",
    "fourierbgc_with_year": "fourierbgc_broadcast",
    "fourierbgc_learned":   "fourierbgc",
    "ppcon_no_scalar":      "ppcon_no_coord",
}


def resolve(name):
    """Canonical registry key for a paper name, registry key, or legacy flag."""
    return LEGACY_NAMES.get(name, name)


def make_model(name, **kwargs):
    key = resolve(name)
    if key == "cnn_mlp_coord_norm":
        return CNNMLPCoord(normalize=True, **kwargs)
    if key not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key](**kwargs)
