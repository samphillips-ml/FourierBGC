"""
Trains the models in the ablation spine under the common recipe: Adam at
1e-3, five-epoch linear warmup then cosine decay, gradient clipping at max
norm 1.0, 100 epochs, batch size 32.

Covers CNN-NoCoord, CNN-RawCoord, CNN-MLPCoord and FourierBGC-Broadcast.
The other two models have their own entry points, because their training
differs: FourierBGC computes its Fourier projection inside the model
(scripts/train_fourierbgc.py), and PPCon-NoCoord uses PPCon's own recipe
rather than this one (scripts/train_ppcon_no_coord.py). PPCon itself is
never retrained; its released checkpoint is evaluated directly.

    python train.py --model cnn_no_coord         --target_var NITRATE --seed 0
    python train.py --model cnn_raw_coord        --target_var CHLA    --seed 0
    python train.py --model cnn_mlp_coord        --target_var BBP700  --seed 0
    python train.py --model fourierbgc_broadcast --target_var NITRATE --seed 0

Checkpoints land at results/{model}/{VAR}/ unless --save_dir overrides it.
"""
import argparse
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from helpers.dataset import FloatDataset
from helpers.fourier_features import compute_fourier_features
from helpers.scalar_norm import normalize_scalars, SCALAR_STATS
from helpers.seeding import set_seed, seeded_generator
from models import make_model, resolve

# PPCon's own depth grids (dict.py): nitrate 0-1000m @ 5m, chla/bbp700 0-200m @ 1m.
# both land at 200 points but they're physically different distances.
DEPTH_GRIDS = {
    "NITRATE": torch.arange(0, 1000, 5).float(),
    "CHLA":    torch.arange(0, 200, 1).float(),
    "BBP700":  torch.arange(0, 200, 1).float(),
}

DATA_DIR = "data"


def run_epoch(model, loader, depth_levels, device, optimizer=None, max_grad_norm=1.0,
              use_scalars=False, use_fourier=False, use_fourier_year=False,
              use_mlpcoord=False, use_flp=False):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    n_depth = depth_levels.shape[0]
    losses = []
    with torch.set_grad_enabled(train_mode):
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            # raw probe: T/S/O only, geolocation/date dropped on purpose
            profile = torch.stack([temp, psal, doxy], dim=-1).to(device)  # (B, D, 3)
            target = target.unsqueeze(-1).to(device)                      # (B, D, 1)

            scalars = None
            if use_scalars:
                # broadcast each (z-scored) scalar to a constant-valued depth
                # channel; the model concatenates these after its backbone,
                # not at the input (see models/cnn_raw_coord.py for why).
                b = profile.shape[0]
                lat, lon, day_rad, year = normalize_scalars(lat, lon, day_rad, year)
                scalars = torch.stack([lat, lon, day_rad, year], dim=-1).to(device)  # (B, 4)
                scalars = scalars.view(b, 1, 4).expand(b, n_depth, 4)  # (B, D, 4)
            elif use_fourier:
                # bounded Fourier encoding of lat/lon/day_of_year, fused at
                # the input alongside T/S/O (see helpers/fourier_features.py). year is not used.
                b = profile.shape[0]
                day_rad, lat, lon = day_rad.to(device), lat.to(device), lon.to(device)
                fourier = compute_fourier_features(day_rad, lat, lon)  # (B, 18)
                fourier = fourier.view(b, 18, 1).expand(b, 18, n_depth).transpose(1, 2)  # (B, D, 18)
                profile = torch.cat([profile, fourier], dim=-1)  # (B, D, 21)
            elif use_fourier_year:
                # exploratory FourierBGCBroadcast variant: FourierBGC_Broadcast_NoYear's 21
                # channels plus one raw (non-Fourier-encoded), z-scored year
                # channel broadcast across depth, fused at the input, 22
                # total. See models/fourierbgc_broadcast.py.
                b = profile.shape[0]
                day_rad, lat, lon = day_rad.to(device), lat.to(device), lon.to(device)
                year = year.to(device)
                fourier = compute_fourier_features(day_rad, lat, lon)  # (B, 18)
                fourier = fourier.view(b, 18, 1).expand(b, 18, n_depth).transpose(1, 2)  # (B, D, 18)
                year_mean, year_std = SCALAR_STATS["year"]
                year_z = (year - year_mean) / year_std
                year_ch = year_z.view(b, 1, 1).expand(b, n_depth, 1)  # (B, D, 1)
                profile = torch.cat([profile, fourier, year_ch], dim=-1)  # (B, D, 22)

            if use_mlpcoord or use_flp:
                # Both take the four raw coordinates and encode them inside the
                # model: CNN-MLPCoord through four per-coordinate MLPs,
                # FourierBGC through the fixed Fourier basis plus a projection.
                # They share a signature, so one branch serves both.
                output = model(profile, depth_levels,
                               day_rad.to(device), year.to(device),
                               lat.to(device), lon.to(device))
            else:
                output = model(profile, depth_levels, scalars)
            loss = mse_loss(output, target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                # caps any single batch's gradient norm, so one bad batch
                # can't throw a destabilizing step (this is what epoch 68's
                # spike in an earlier run looked like)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            losses.append(loss.item())

    return np.mean(losses)


def make_scheduler(optimizer, total_epochs, warmup_epochs=5):
    # linear warmup for warmup_epochs, then cosine decay to 0 over the rest.
    # standard transformer recipe, not PPCon-specific, no fidelity concern.
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=["cnn_no_coord", "cnn_raw_coord", "cnn_mlp_coord",
                            "fourierbgc_broadcast", "fourierbgc", "cnn_mlp_coord_norm",
                            # legacy aliases, kept so old commands keep working
                            "cnn", "cnn_scalar", "cnn_mlpcoord", "fourierbgc_with_year",
                            "fourierbgc_learned"],
                   required=True,
                   help="see models/__init__.py for the paper-name mapping")
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--save_dir", default=None,
                    help="override the computed save_dir (results_dir/target_var/model); "
                         "used to route exploratory variants to their own directory tree")
    p.add_argument("--seed", type=int, default=None,
                    help="optional seed for reproducibility: seeds python/numpy/torch RNG "
                         "and the train DataLoader's shuffle order. Default None preserves "
                         "the old unseeded behavior exactly (every existing single-run model "
                         "in this repo was trained this way).")
    args = p.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.backends.mps.is_available():
     #   device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}, model: {args.model}, target: {args.target_var}, seed: {args.seed}")

    train_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_train.csv"))
    test_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_test.csv"))
    train_generator = seeded_generator(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=train_generator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(args.model).to(device)
    key = resolve(args.model)
    use_scalars = key == "cnn_raw_coord"
    use_fourier = False  # the 21-channel no-year variant is not in the manuscript
    use_fourier_year = key == "fourierbgc_broadcast"
    use_mlpcoord = key in ("cnn_mlp_coord", "cnn_mlp_coord_norm")
    use_flp = key == "fourierbgc"
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs)

    if args.save_dir:
        save_dir = args.save_dir
    elif args.seed is not None:
        save_dir = os.path.join(args.results_dir, key, f"seed{args.seed}", args.target_var)
    else:
        save_dir = os.path.join(args.results_dir, key, args.target_var)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    best_test_mse = float("inf")
    with open(log_path, "w") as log:
        for ep in range(args.epochs):
            train_mse = run_epoch(model, train_loader, depth_levels, device, optimizer,
                                   use_scalars=use_scalars, use_fourier=use_fourier,
                                   use_fourier_year=use_fourier_year,
                                   use_mlpcoord=use_mlpcoord, use_flp=use_flp)
            test_mse = run_epoch(model, test_loader, depth_levels, device, optimizer=None,
                                  use_scalars=use_scalars, use_fourier=use_fourier,
                                  use_fourier_year=use_fourier_year,
                                  use_mlpcoord=use_mlpcoord, use_flp=use_flp)
            scheduler.step()

            line = (f"epoch {ep+1:4d}  train_mse {train_mse:.5f}  test_mse {test_mse:.5f}"
                    f"  lr {scheduler.get_last_lr()[0]:.6f}")
            print(line)
            log.write(line + "\n")

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(save_dir, "final.pt"))
    print(f"best test_mse: {best_test_mse:.5f}, saved to {save_dir}/best.pt")


if __name__ == "__main__":
    main()