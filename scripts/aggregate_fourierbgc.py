"""
Aggregates the 5-seed FourierBGC (learned-projection-with-year) results for
one target variable: runs run_evaluate_with_year.py's per_profile_rmse
against each seed's best.pt checkpoint, reports overall RMSE as mean +/- SD
across seeds and per-region RMSE (mean +/- SD across seeds), alongside the
locked comparison numbers (FourierBGC-Broadcast and PPCon).

SUPERSEDED LABELING NOTE: the LOCKED_* numbers below were historically
mislabeled "FourierBGC (=WithYear)" in this script, but they are actually
FourierBGCBroadcast's (models.py's fixed/non-learned-projection variant)
5-seed numbers, not this script's own FourierBGC (learned-projection)
class. Labels corrected below to "FourierBGC-Broadcast"; values are
unchanged and were already correct as FourierBGC-Broadcast's numbers.

Locked FourierBGC-Broadcast numbers below were recomputed directly from
~/Dev/xuyang-li/seeded_ablation_results/fourierbgc_with_year/seed{0-4}/
{VAR}/eval_best.txt (the actual 5-seed sweep backing the paper's
FourierBGC-Broadcast numbers); CHLA matches the published table exactly.
PPCon numbers are from decisions.md's confirmed-results table (single
checkpoint, not re-derivable here).

    python fourier_learned_projection/aggregate_results_with_year.py --target_var CHLA
    python fourier_learned_projection/aggregate_results_with_year.py --target_var NITRATE
    python fourier_learned_projection/aggregate_results_with_year.py --target_var BBP700
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helpers.dataset import FloatDataset  # noqa: E402
from train import DEPTH_GRIDS  # noqa: E402
from evaluate import REGIONS  # noqa: E402

sys.path.insert(0, str(_HERE))
from models import FourierBGC  # noqa: E402
from run_evaluate_with_year import per_profile_rmse, region_rmse  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]

LOCKED_OVERALL = {
    "CHLA":    {"FourierBGC-Broadcast": (0.0723, 0.0009), "PPCon": (0.0802, None)},
    "NITRATE": {"FourierBGC-Broadcast": (0.2288, 0.0035), "PPCon": (0.5234, None)},
    "BBP700":  {"FourierBGC-Broadcast": (0.0002, 0.0000), "PPCon": (0.0002, None)},
}
LOCKED_REGION = {
    "CHLA": {
        "NWM": {"FourierBGC-Broadcast": (0.1035, 0.0021), "PPCon": (0.1299, None)},
        "SWM": {"FourierBGC-Broadcast": (0.1188, 0.0018), "PPCon": (0.1187, None)},
        "TYR": {"FourierBGC-Broadcast": (0.0808, 0.0011), "PPCon": (0.0840, None)},
        "ION": {"FourierBGC-Broadcast": (0.0430, 0.0006), "PPCon": (0.0445, None)},
        "LEV": {"FourierBGC-Broadcast": (0.0512, 0.0007), "PPCon": (0.0468, None)},
    },
    "NITRATE": {
        "NWM": {"FourierBGC-Broadcast": (0.2654, 0.0026), "PPCon": (0.6532, None)},
        "SWM": {"FourierBGC-Broadcast": (0.2169, 0.0101), "PPCon": (0.5014, None)},
        "TYR": {"FourierBGC-Broadcast": (0.1722, 0.0077), "PPCon": (0.4408, None)},
        "ION": {"FourierBGC-Broadcast": (0.1761, 0.0032), "PPCon": (0.4121, None)},
        "LEV": {"FourierBGC-Broadcast": (0.2441, 0.0088), "PPCon": (0.5143, None)},
    },
    "BBP700": {
        "NWM": {"FourierBGC-Broadcast": (0.0002, 0.0000), "PPCon": (0.0002, None)},
        "SWM": {"FourierBGC-Broadcast": (0.0002, 0.0000), "PPCon": (0.0002, None)},
        "TYR": {"FourierBGC-Broadcast": (0.0002, 0.0000), "PPCon": (0.0002, None)},
        "ION": {"FourierBGC-Broadcast": (0.0001, 0.0000), "PPCon": (0.0002, None)},
        "LEV": {"FourierBGC-Broadcast": (0.0001, 0.0000), "PPCon": (0.0002, None)},
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], default="CHLA")
    p.add_argument("--results_dir", default=None,
                    help="default: fourier_learned_projection/results_with_year/{target_var}")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    test_ds = FloatDataset(str(_REPO_ROOT / "data" / args.target_var / "float_ds_sf_test.csv"))
    results_dir = Path(args.results_dir) if args.results_dir else (
        _HERE / "results_with_year" / args.target_var)

    overall_per_seed = []
    region_per_seed = {name: [] for name in REGIONS}
    n_per_region = {}

    for seed in SEEDS:
        ckpt = results_dir / f"seed{seed}" / "best.pt"
        model = FourierBGC().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        records = per_profile_rmse(model, test_ds, depth_levels, device, args.target_var)
        overall = float(np.mean([r["rmse"] for r in records]))
        overall_per_seed.append(overall)

        for name, (val, n) in region_rmse(records).items():
            if val is not None:
                region_per_seed[name].append(val)
                n_per_region[name] = n

        print(f"seed {seed}: overall {overall:.6g}")

    overall_arr = np.array(overall_per_seed)
    print(f"\nFourierBGC {args.target_var} overall test RMSE: "
          f"{overall_arr.mean():.6g} +/- {overall_arr.std(ddof=1):.6g}  (n={len(SEEDS)} seeds)")

    print("\nLocked comparison numbers:")
    for name, (mean, sd) in LOCKED_OVERALL[args.target_var].items():
        sd_str = f" +/- {sd:.6g}" if sd is not None else ""
        print(f"  {name:24s}: {mean:.6g}{sd_str}")

    print("\nper-region RMSE (mean +/- SD across seeds):")
    for name in REGIONS:
        vals = region_per_seed[name]
        if not vals:
            print(f"  {name:4s}  no samples in this region")
            continue
        arr = np.array(vals)
        n = n_per_region.get(name, "?")
        print(f"  {name:4s}  learned-proj {arr.mean():.6g} +/- {arr.std(ddof=1):.6g}   n={n}")
        for cname, (cmean, csd) in LOCKED_REGION[args.target_var][name].items():
            csd_str = f" +/- {csd:.6g}" if csd is not None else ""
            print(f"        {cname:24s}: {cmean:.6g}{csd_str}")


if __name__ == "__main__":
    main()
