"""
Evaluates a trained FourierBGC checkpoint. Same
per-profile RMSE methodology as run_evaluate.py / evaluate.py (REGIONS,
assign_region, day_rad_to_day imported, not reimplemented); the only
difference is year is pulled off the loader and passed into the model.

--target_var defaults to CHLA. For NITRATE, matches evaluate.py's
per_profile_rmse: drops the deepest 10 points before computing RMSE. For
BBP700, matches evaluate.py: divides both pred and true by 1000 first
(stored values are x1000). Both quirks copied from evaluate.py, not
reimplemented independently, so results stay directly comparable to every
other model's regional/seasonal table in this repo.

    python scripts/evaluate_fourierbgc.py \
        --checkpoint fourier_learned_projection/results_with_year/CHLA/seed0/best.pt
    python scripts/evaluate_fourierbgc.py --target_var NITRATE \
        --checkpoint fourier_learned_projection/results_with_year/NITRATE/seed0/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from helpers.dataset import FloatDataset  # noqa: E402
from train import DEPTH_GRIDS  # noqa: E402
from evaluate import REGIONS, assign_region, day_rad_to_day  # noqa: E402

from models import FourierBGC  # noqa: E402

DATA_DIR = str(Path(_REPO_ROOT) / "data")


def per_profile_rmse(model, dataset, depth_levels, device, target_var):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()

    records = []
    with torch.no_grad():
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            profile = torch.stack([temp, psal, doxy], dim=-1).to(device)
            day_rad_d, lat_d, lon_d = day_rad.to(device), lat.to(device), lon.to(device)
            year_d = year.to(device)
            pred = model(profile, depth_levels, day_rad_d, lat_d, lon_d, year_d).squeeze()
            true = target.squeeze().to(device)

            if target_var == "NITRATE":
                # matches evaluate.py's per_profile_rmse / PPCon's get_reconstruction
                pred = pred[:-10]
                true = true[:-10]
            elif target_var == "BBP700":
                # matches evaluate.py: stored values are x1000
                pred = pred / 1000
                true = true / 1000

            rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
            records.append({
                "rmse": rmse,
                "lat": lat.item(),
                "lon": lon.item(),
                "day": day_rad_to_day(day_rad.item()),
            })
    return records


def region_rmse(records):
    out = {}
    for name in REGIONS:
        vals = [r["rmse"] for r in records if assign_region(r["lat"], r["lon"]) == name]
        out[name] = (float(np.mean(vals)) if vals else None, len(vals))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], default="CHLA")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default=DATA_DIR)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    test_ds = FloatDataset(f"{args.data_dir}/{args.target_var}/float_ds_sf_test.csv")

    model = FourierBGC().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    records = per_profile_rmse(model, test_ds, depth_levels, device, args.target_var)
    overall = float(np.mean([r["rmse"] for r in records]))
    print(f"overall test RMSE: {overall:.4f}  (n={len(records)})")
    for name, (val, n) in region_rmse(records).items():
        if val is None:
            print(f"  {name:4s}  no samples in this region")
        else:
            print(f"  {name:4s}  rmse {val:.4f}   n={n}")


if __name__ == "__main__":
    main()
