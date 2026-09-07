"""
Loads PPCon's released checkpoints and reproduces their forward pass for
evaluate.py. Nothing in third_party/ppcon/ is modified.
"""
import os
import sys

import torch

_PPCON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "ppcon")
if _PPCON_DIR not in sys.path:
    sys.path.insert(0, _PPCON_DIR)

from ppcon.train.conv1med_dp import Conv1dMed  # noqa: E402  # type: ignore[import]
from ppcon.train.mlp import MLPDay, MLPYear, MLPLat, MLPLon  # noqa: E402  # type: ignore[import]


def load_ppcon_checkpoint(model_dir, epoch, device, dp_rate=0.2):
    """Loads the five state_dicts PPCon's training saves per snapshot
    (model_day_{epoch}.pt, model_year_{epoch}.pt, model_lat_{epoch}.pt,
    model_lon_{epoch}.pt, model_conv_{epoch}.pt) and returns the five models
    in eval mode. dp_rate only affects Conv1dMed's Dropout modules, which are
    inactive in eval mode either way, so its value has no effect on results."""
    model_day = MLPDay().to(device)
    model_year = MLPYear().to(device)
    model_lat = MLPLat().to(device)
    model_lon = MLPLon().to(device)
    model_conv = Conv1dMed(dp_rate=dp_rate).to(device)

    model_day.load_state_dict(torch.load(os.path.join(model_dir, f"model_day_{epoch}.pt"), map_location=device))
    model_year.load_state_dict(torch.load(os.path.join(model_dir, f"model_year_{epoch}.pt"), map_location=device))
    model_lat.load_state_dict(torch.load(os.path.join(model_dir, f"model_lat_{epoch}.pt"), map_location=device))
    model_lon.load_state_dict(torch.load(os.path.join(model_dir, f"model_lon_{epoch}.pt"), map_location=device))
    model_conv.load_state_dict(torch.load(os.path.join(model_dir, f"model_conv_{epoch}.pt"), map_location=device))

    for m in (model_day, model_year, model_lat, model_lon, model_conv):
        m.eval()

    return model_day, model_year, model_lat, model_lon, model_conv


def ppcon_forward(models, year, day, lat, lon, temp, psal, doxy):
    """Reproduces train.py's train_model concatenation exactly. year/day/lat/lon
    are (B,) tensors, temp/psal/doxy are (B, D) tensors. Returns the raw
    Conv1dMed output, shape (B, 1, D) -- squeeze() collapses this to (D,) for
    batch_size=1 callers, same as the models in models/'s (B, D, 1)."""
    model_day, model_year, model_lat, model_lon, model_conv = models

    output_day = model_day(day.unsqueeze(1))
    output_year = model_year(year.unsqueeze(1).float())
    output_lat = model_lat(lat.unsqueeze(1).float())
    output_lon = model_lon(lon.unsqueeze(1).float())

    output_day = torch.transpose(output_day.unsqueeze(0), 0, 1)
    output_year = torch.transpose(output_year.unsqueeze(0), 0, 1)
    output_lat = torch.transpose(output_lat.unsqueeze(0), 0, 1)
    output_lon = torch.transpose(output_lon.unsqueeze(0), 0, 1)
    temp = torch.transpose(temp.unsqueeze(0), 0, 1)
    psal = torch.transpose(psal.unsqueeze(0), 0, 1)
    doxy = torch.transpose(doxy.unsqueeze(0), 0, 1)

    x = torch.cat((output_day, output_year, output_lat, output_lon, temp, psal, doxy), 1)
    return model_conv(x.float())
