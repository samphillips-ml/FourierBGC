"""
Reproduces PPCon's own RMSE methodology, from analysis/rmse.py and
analysis/utils_analysis.py in github.com/gpietrop/PPCon (MIT). Per-profile
RMSE (sqrt of that profile's own MSE across all 200 depth points), then
averaged within region and season buckets, weighted by sample count, not
pooled across all points first. Also reports a single overall number
(per-profile RMSE averaged across the whole test set, no bucketing), which
is the form Appendix B's Table B1 number (0.52 for nitrate) is in.

    python evaluate.py --model transformer --target_var NITRATE \
        --checkpoint results/NITRATE/transformer/best.pt

    python evaluate.py --model ppcon --target_var NITRATE \
        --checkpoint_dir results_ppcon/NITRATE/2024-01-01/model --epoch 200
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FloatDataset
from models import RawTransformerProbe, RawCNNProbe
from ppcon_eval import load_ppcon_checkpoint, ppcon_forward
from train import DEPTH_GRIDS, make_model

# from utils_analysis.py, dict_ga: [[lat_min, lat_max], [lon_min, lon_max]]
REGIONS = {
    "NWM": [[40, 45], [-2, 9.5]],
    "SWM": [[32, 40], [-2, 9.5]],
    "TYR": [[37, 45], [9.5, 16]],
    "ION": [[30, 37], [9.5, 22]],
    "LEV": [[30, 37], [22, 36]],
}

# from utils_analysis.py, dict_season: day-of-year ranges
SEASONS = {
    "W": [0, 91],
    "SP": [92, 182],
    "SU": [183, 273],
    "A": [274, 365],
}


def day_rad_to_day(day_rad):
    # from utils_train.py: from_day_rad_to_day
    return (day_rad * 365) / (2 * np.pi)


def assign_region(lat, lon):
    for name, ((lat_min, lat_max), (lon_min, lon_max)) in REGIONS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return None  # falls outside all five boxes, excluded from regional table


def assign_season(day):
    for name, (lo, hi) in SEASONS.items():
        if lo <= day <= hi:
            return name
    return None


def per_profile_rmse(model, dataset, depth_levels, target_var, device, is_ppcon=False):
    """Runs every profile through the model one at a time (matches PPCon's
    own get_reconstruction, which also iterates with shuffle and no batching),
    returns per-profile RMSE plus the lat/lon/season needed for bucketing.
    For the PPCon baseline (is_ppcon=True), `model` is the five-model tuple
    from load_ppcon_checkpoint and the forward pass goes through ppcon_forward
    instead of the RawCNNProbe/RawTransformerProbe call."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if not is_ppcon:
        model.eval()

    records = []
    with torch.no_grad():
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            if is_ppcon:
                year, day_rad = year.to(device), day_rad.to(device)
                lat, lon = lat.to(device), lon.to(device)
                temp, psal, doxy = temp.to(device), psal.to(device), doxy.to(device)
                pred = ppcon_forward(model, year, day_rad, lat, lon, temp, psal, doxy).squeeze()
            else:
                profile = torch.stack([temp, psal, doxy], dim=-1).to(device)
                pred = model(profile, depth_levels).squeeze()      # (200,)
            true = target.squeeze().to(device)                  # (200,)

            if target_var == "NITRATE":
                # matches get_reconstruction: drops the deepest 10 points
                pred = pred[:-10]
                true = true[:-10]
            elif target_var == "BBP700":
                # matches get_reconstruction: stored values are x1000
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


def summarize(records):
    overall = np.mean([r["rmse"] for r in records])
    print(f"\noverall test RMSE (per-profile, pooled across whole test set): {overall:.4f}")
    print("(compare directly to PPCon's Appendix B Table B1 number)\n")

    print("by region:")
    for name in REGIONS:
        vals = [r["rmse"] for r in records if assign_region(r["lat"], r["lon"]) == name]
        if vals:
            print(f"  {name:4s}  rmse {np.mean(vals):.4f}   n={len(vals)}")
        else:
            print(f"  {name:4s}  no samples in this region")

    print("\nby season:")
    for name in SEASONS:
        vals = [r["rmse"] for r in records if assign_season(r["day"]) == name]
        if vals:
            print(f"  {name:4s}  rmse {np.mean(vals):.4f}   n={len(vals)}")
        else:
            print(f"  {name:4s}  no samples in this season")

    n_unassigned = sum(1 for r in records if assign_region(r["lat"], r["lon"]) is None)
    if n_unassigned:
        print(f"\nnote: {n_unassigned}/{len(records)} profiles fall outside all five region "
              f"boxes, excluded from the regional breakdown (but included in 'overall')")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["transformer", "cnn", "ppcon"], required=True)
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], required=True)
    p.add_argument("--checkpoint", help="required for --model transformer/cnn")
    p.add_argument("--checkpoint_dir", help="required for --model ppcon: the .../model/ "
                   "directory holding model_{day,year,lat,lon,conv}_{epoch}.pt")
    p.add_argument("--epoch", type=int, help="required for --model ppcon")
    p.add_argument("--dp_rate", type=float, default=0.2,
                   help="ppcon Conv1dMed dropout rate; inactive in eval mode, value has no effect")
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()

    if args.model == "ppcon":
        if not args.checkpoint_dir or args.epoch is None:
            p.error("--model ppcon requires --checkpoint_dir and --epoch")
    elif not args.checkpoint:
        p.error("--model transformer/cnn requires --checkpoint")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    test_ds = FloatDataset(f"{args.data_dir}/{args.target_var}/float_ds_sf_test.csv")

    if args.model == "ppcon":
        model = load_ppcon_checkpoint(args.checkpoint_dir, args.epoch, device, dp_rate=args.dp_rate)
        records = per_profile_rmse(model, test_ds, depth_levels, args.target_var, device, is_ppcon=True)
    else:
        model = make_model(args.model).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        records = per_profile_rmse(model, test_ds, depth_levels, args.target_var, device)

    summarize(records)


if __name__ == "__main__":
    main()