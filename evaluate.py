"""
Reproduces PPCon's own RMSE methodology, from analysis/rmse.py and
analysis/utils_analysis.py in github.com/gpietrop/PPCon (MIT). Per-profile
RMSE (sqrt of that profile's own MSE across all 200 depth points), then
averaged within region and season buckets, weighted by sample count, not
pooled across all points first. Also reports a single overall number
(per-profile RMSE averaged across the whole test set, no bucketing).
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from helpers.dataset import FloatDataset
from helpers.fourier_features import compute_fourier_features
from helpers.ppcon_eval import load_ppcon_checkpoint, ppcon_forward
from helpers.scalar_norm import normalize_scalars, SCALAR_STATS
from models import make_model, resolve
from models.ppcon_no_coord import (load_ppcon_no_coord_checkpoint,
                                   ppcon_no_coord_forward)
from train import DEPTH_GRIDS

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


def per_profile_rmse(model, dataset, depth_levels, target_var, device, is_ppcon=False,
                      is_ppcon_no_scalar=False, use_scalars=False, use_fourier=False,
                      use_fourier_year=False, use_mlpcoord=False, use_flp=False):
    """Runs every profile through the model one at a time (matches PPCon's
    own get_reconstruction, which also iterates with shuffle and no batching),
    returns per-profile RMSE plus the lat/lon/season needed for bucketing.
    For the PPCon baseline (is_ppcon=True), `model` is the five-model tuple
    from load_ppcon_checkpoint and the forward pass goes through ppcon_forward
    instead of the RawCNNProbe/RawCNNScalarProbe/RawTransformerProbe/
    RawTransformerScalarProbe/FourierBGC_Broadcast_NoYear call. is_ppcon_no_scalar=True is the
    ablation: `model` is the single Conv1dMed from
    load_ppcon_no_scalar_checkpoint and the forward pass goes through
    ppcon_no_scalar_forward, which takes only temp/psal/doxy -- no scalar
    inputs at all. use_scalars=True (transformer_scalar/cnn_scalar) broadcasts
    (z-scored) lat/lon/day_rad/year to constant-valued depth channels and
    passes them to the model as a separate `scalars` argument, concatenated
    after the backbone, not at the input, same as train.py's run_epoch.
    use_fourier=True (fourierbgc) instead concatenates a bounded Fourier
    encoding of lat/lon/day_of_year onto T/S/O at the input (see
    fourier_features.py); year is not used."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    if not is_ppcon and not is_ppcon_no_scalar:
        model.eval()

    n_depth = depth_levels.shape[0]
    records = []
    with torch.no_grad():
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            if is_ppcon:
                year, day_rad = year.to(device), day_rad.to(device)
                lat, lon = lat.to(device), lon.to(device)
                temp, psal, doxy = temp.to(device), psal.to(device), doxy.to(device)
                pred = ppcon_forward(model, year, day_rad, lat, lon, temp, psal, doxy).squeeze()
            elif is_ppcon_no_scalar:
                temp, psal, doxy = temp.to(device), psal.to(device), doxy.to(device)
                pred = ppcon_no_coord_forward(model, temp, psal, doxy).squeeze()
            else:
                profile = torch.stack([temp, psal, doxy], dim=-1).to(device)
                scalars = None
                if use_scalars:
                    b = profile.shape[0]
                    n_lat, n_lon, n_day_rad, n_year = normalize_scalars(lat, lon, day_rad, year)
                    scalars = torch.stack([n_lat, n_lon, n_day_rad, n_year], dim=-1).to(device)
                    scalars = scalars.view(b, 1, 4).expand(b, n_depth, 4)
                elif use_fourier:
                    b = profile.shape[0]
                    day_rad_d, lat_d, lon_d = day_rad.to(device), lat.to(device), lon.to(device)
                    fourier = compute_fourier_features(day_rad_d, lat_d, lon_d)  # (B, 18)
                    fourier = fourier.view(b, 18, 1).expand(b, 18, n_depth).transpose(1, 2)
                    profile = torch.cat([profile, fourier], dim=-1)  # (B, D, 21)
                elif use_fourier_year:
                    b = profile.shape[0]
                    day_rad_d, lat_d, lon_d = day_rad.to(device), lat.to(device), lon.to(device)
                    year_d = year.to(device)
                    fourier = compute_fourier_features(day_rad_d, lat_d, lon_d)  # (B, 18)
                    fourier = fourier.view(b, 18, 1).expand(b, 18, n_depth).transpose(1, 2)
                    year_mean, year_std = SCALAR_STATS["year"]
                    year_z = (year_d - year_mean) / year_std
                    year_ch = year_z.view(b, 1, 1).expand(b, n_depth, 1)  # (B, D, 1)
                    profile = torch.cat([profile, fourier, year_ch], dim=-1)  # (B, D, 22)
                if use_flp:
                    pred = model(profile, depth_levels,
                                 day_rad.to(device), year.to(device),
                                 lat.to(device), lon.to(device)).squeeze()
                elif use_mlpcoord:
                    # PPCon's own encoder signature; four raw scalars, each
                    # through its own MLP inside the model.
                    pred = model(profile, depth_levels,
                                 day_rad.to(device), year.to(device),
                                 lat.to(device), lon.to(device)).squeeze()
                else:
                    pred = model(profile, depth_levels, scalars).squeeze()  # (200,)
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
    print(f"\noverall test RMSE (per-profile, pooled across whole test set): {overall:.6g}")
    print("(compare directly to PPCon's Appendix B Table B1 number)\n")

    print("by region:")
    for name in REGIONS:
        vals = [r["rmse"] for r in records if assign_region(r["lat"], r["lon"]) == name]
        if vals:
            print(f"  {name:4s}  rmse {np.mean(vals):.6g}   n={len(vals)}")
        else:
            print(f"  {name:4s}  no samples in this region")

    print("\nby season:")
    for name in SEASONS:
        vals = [r["rmse"] for r in records if assign_season(r["day"]) == name]
        if vals:
            print(f"  {name:4s}  rmse {np.mean(vals):.6g}   n={len(vals)}")
        else:
            print(f"  {name:4s}  no samples in this season")

    n_unassigned = sum(1 for r in records if assign_region(r["lat"], r["lon"]) is None)
    if n_unassigned:
        print(f"\nnote: {n_unassigned}/{len(records)} profiles fall outside all five region "
              f"boxes, excluded from the regional breakdown (but included in 'overall')")


def evaluate_checkpoint(key, target_var, test_ds, depth_levels, device,
                        checkpoint=None, checkpoint_dir=None, epoch=None, dp_rate=0.2):
    """Loads one checkpoint and returns its per-profile records."""
    if key == "ppcon":
        model = load_ppcon_checkpoint(checkpoint_dir, epoch, device, dp_rate=dp_rate)
        return per_profile_rmse(model, test_ds, depth_levels, target_var, device, is_ppcon=True)
    if key == "ppcon_no_coord":
        model = load_ppcon_no_coord_checkpoint(checkpoint_dir, epoch, device, dp_rate=dp_rate)
        return per_profile_rmse(model, test_ds, depth_levels, target_var, device,
                                is_ppcon_no_scalar=True)
    model = make_model(key).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return per_profile_rmse(
        model, test_ds, depth_levels, target_var, device,
        use_scalars=(key == "cnn_raw_coord"),
        use_fourier=False,
        use_fourier_year=(key == "fourierbgc_broadcast"),
        use_mlpcoord=(key in ("cnn_mlp_coord", "cnn_mlp_coord_norm")),
        use_flp=(key == "fourierbgc"))


def summarize_seeds(per_seed_records):
    """Mean and standard deviation across seeds, for the pooled figure and for
    every region and season bucket. This is the form the manuscript's tables
    report, so it is what you want when checking a published number."""
    def bucket(records, name):
        if name == "overall":
            return [r["rmse"] for r in records]
        if name in REGIONS:
            return [r["rmse"] for r in records if assign_region(r["lat"], r["lon"]) == name]
        return [r["rmse"] for r in records if assign_season(r["day"]) == name]

    n = len(per_seed_records)
    print(f"\nacross {n} seeds, mean (sd):\n")
    for name in ["overall"] + list(REGIONS) + list(SEASONS):
        means = [np.mean(v) for v in (bucket(rec, name) for rec in per_seed_records) if len(v)]
        if not means:
            continue
        mu = float(np.mean(means))
        sd = float(np.std(means, ddof=1)) if n > 1 else 0.0
        label = "pooled" if name == "overall" else name
        count = len(bucket(per_seed_records[0], name))
        print(f"  {label:8s} {mu:.6g}  (sd {sd:.3g})   n={count}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["cnn_no_coord", "cnn_raw_coord", "cnn_mlp_coord",
                            "fourierbgc_broadcast", "fourierbgc",
                            "ppcon", "ppcon_no_coord", "cnn_mlp_coord_norm",
                            # legacy aliases, kept so old commands keep working
                            "cnn", "cnn_scalar", "cnn_mlpcoord",
                            "fourierbgc_with_year", "fourierbgc_learned",
                            "ppcon_no_scalar"],
                   required=True)
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], required=True)
    p.add_argument("--checkpoint", help="required for --model transformer/cnn")
    p.add_argument("--checkpoint_dir", help="required for --model ppcon/ppcon_no_scalar: the "
                   ".../model/ directory holding model_{day,year,lat,lon,conv}_{epoch}.pt "
                   "(ppcon) or model_conv_{epoch}.pt (ppcon_no_scalar)")
    p.add_argument("--epoch", type=int, help="required for --model ppcon/ppcon_no_scalar")
    p.add_argument("--dp_rate", type=float, default=0.2,
                   help="ppcon/ppcon_no_scalar Conv1dMed dropout rate; inactive in eval mode, "
                   "value has no effect")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds, e.g. 0,1,2,3,4. Evaluates every seed and "
                        "reports mean and sd across them, deriving checkpoint paths from "
                        "--results_dir. Overrides --checkpoint/--checkpoint_dir.")
    p.add_argument("--results_dir", default="results",
                   help="root for --seeds path derivation (default: results)")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None

    if seeds is not None:
        if resolve(args.model) == "ppcon":
            p.error("--model ppcon has no seeds; it is a single released checkpoint")
        if resolve(args.model) == "ppcon_no_coord" and args.epoch is None:
            p.error("--model ppcon_no_coord with --seeds still requires --epoch")
    elif resolve(args.model) in ("ppcon", "ppcon_no_coord"):
        if not args.checkpoint_dir or args.epoch is None:
            p.error(f"--model {args.model} requires --checkpoint_dir and --epoch")
    elif not args.checkpoint:
        p.error(f"--model {args.model} requires --checkpoint (or --seeds)")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    test_ds = FloatDataset(f"{args.data_dir}/{args.target_var}/float_ds_sf_test.csv")

    key = resolve(args.model)

    if seeds is not None:
        per_seed = []
        for sd in seeds:
            base = os.path.join(args.results_dir, key, f"seed{sd}", args.target_var)
            if key == "ppcon_no_coord":
                per_seed.append(evaluate_checkpoint(
                    key, args.target_var, test_ds, depth_levels, device,
                    checkpoint_dir=os.path.join(base, "model"), epoch=args.epoch,
                    dp_rate=args.dp_rate))
            else:
                per_seed.append(evaluate_checkpoint(
                    key, args.target_var, test_ds, depth_levels, device,
                    checkpoint=os.path.join(base, "best.pt")))
            print(f"  seed {sd}: done", flush=True)
        summarize_seeds(per_seed)
        return

    records = evaluate_checkpoint(
        key, args.target_var, test_ds, depth_levels, device,
        checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir,
        epoch=args.epoch, dp_rate=args.dp_rate)

    summarize(records)


if __name__ == "__main__":
    main()