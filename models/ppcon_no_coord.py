"""
PPCon-NoCoord: PPCon with all four coordinate encoders removed and nothing put
in their place, keeping PPCon's own training recipe.

C = 3. Architecturally identical to CNN-NoCoord; the two differ only in how
they are trained, which is what makes the pair a clean isolation of the recipe.

It converges poorly -- 1.8265 on nitrate against PPCon's 0.5234 -- and is read
as a control on that recipe rather than as evidence that the coordinates
matter. It is also the only model in the spine with substantial seed-to-seed
variance (43 % of its mean on nitrate, 32 % on bbp700, against 3 % or less
everywhere else), which is what a mismatched regularization strength would
produce: PPCon's loss applies a weight-regularization coefficient tuned across
all 412,049 parameters, and removing the encoders removes 158,800 of them.

There is no class of our own here. The model *is* PPCon's Conv1dMed with
in_channels monkeypatched to 3, so this module holds the loader and the
forward pass rather than a definition. in_channels is a module-level constant
in PPCon's conv1med_dp.py, so it is patched only for the duration of model
construction and then restored, letting this coexist in one process with the
7-channel PPCon loader in helpers/ppcon_eval.py.
"""
import os
import sys
from contextlib import contextmanager

import torch

_PPCON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "ppcon")
if _PPCON_DIR not in sys.path:
    sys.path.insert(0, _PPCON_DIR)

from ppcon.train import conv1med_dp  # noqa: E402  # type: ignore[import]
from ppcon.train.conv1med_dp import Conv1dMed  # noqa: E402  # type: ignore[import]


@contextmanager
def _conv1dmed_in_channels(n):
    orig = conv1med_dp.in_channels
    conv1med_dp.in_channels = n
    try:
        yield
    finally:
        conv1med_dp.in_channels = orig


def load_ppcon_no_coord_checkpoint(model_dir, epoch, device, dp_rate=0.2):
    """Loads model_conv_{epoch}.pt and returns the model in eval mode. dp_rate
    only affects Dropout modules, inactive in eval mode, so it has no effect on
    results."""
    with _conv1dmed_in_channels(3):
        model_conv = Conv1dMed(dp_rate=dp_rate).to(device)

    model_conv.load_state_dict(
        torch.load(os.path.join(model_dir, f"model_conv_{epoch}.pt"), map_location=device))
    model_conv.eval()

    return model_conv


def ppcon_no_coord_forward(model_conv, temp, psal, doxy):
    """Reproduces the training-time concatenation exactly: (temp, psal, doxy)
    along the channel dim, then Conv1dMed. Returns (B, 1, D)."""
    temp = torch.transpose(temp.unsqueeze(0), 0, 1)
    psal = torch.transpose(psal.unsqueeze(0), 0, 1)
    doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

    x = torch.cat((temp, psal, doxy), 1)
    return model_conv(x.float())


# Backwards-compatible aliases for the pre-reorganization names.
load_ppcon_no_scalar_checkpoint = load_ppcon_no_coord_checkpoint
ppcon_no_scalar_forward = ppcon_no_coord_forward
