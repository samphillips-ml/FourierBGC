"""
Loads PPConNoScalar checkpoints (trained by ppcon_no_scalar/run.py) and
reproduces its forward pass for evaluate.py. Mirrors ppcon_eval.py exactly,
minus the four scalar MLPs: only model_conv_{epoch}.pt is loaded, and the
forward pass concatenates just (temp, psal, doxy) -- the same three-channel
order used in ppcon_no_scalar/train.py's training and validation loops --
before passing through Conv1dMed.

Conv1dMed's in_channels is a module-level constant in conv1med_dp.py, so
it's monkeypatched to 3 only for the duration of model construction (see
ppcon_no_scalar/train.py for why), then restored, so this can coexist in
the same process as ppcon_eval.py's 7-channel PPCon loader.
"""
import os
import sys
from contextlib import contextmanager

import torch

_PPCON_BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppcon_baseline")
if _PPCON_BASELINE_DIR not in sys.path:
    sys.path.insert(0, _PPCON_BASELINE_DIR)

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


def load_ppcon_no_scalar_checkpoint(model_dir, epoch, device, dp_rate=0.2):
    """Loads the single state_dict ppcon_no_scalar/train.py's training saves
    per snapshot (model_conv_{epoch}.pt) and returns the model in eval mode.
    dp_rate only affects Conv1dMed's Dropout modules, which are inactive in
    eval mode either way, so its value has no effect on results."""
    with _conv1dmed_in_channels(3):
        model_conv = Conv1dMed(dp_rate=dp_rate).to(device)

    model_conv.load_state_dict(torch.load(os.path.join(model_dir, f"model_conv_{epoch}.pt"), map_location=device))
    model_conv.eval()

    return model_conv


def ppcon_no_scalar_forward(model_conv, temp, psal, doxy):
    """Reproduces ppcon_no_scalar/train.py's concatenation exactly.
    temp/psal/doxy are (B, D) tensors. Returns the raw Conv1dMed output,
    shape (B, 1, D) -- squeeze() collapses this to (D,) for batch_size=1
    callers, same as ppcon_eval.py's ppcon_forward."""
    temp = torch.transpose(temp.unsqueeze(0), 0, 1)
    psal = torch.transpose(psal.unsqueeze(0), 0, 1)
    doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

    x = torch.cat((temp, psal, doxy), 1)
    return model_conv(x.float())
