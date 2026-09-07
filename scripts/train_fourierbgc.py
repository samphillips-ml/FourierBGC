"""
Trains FourierBGC. Same recipe as train.py's common recipe
(Adam lr=1e-3, batch_size=32, epochs=100, linear-warmup(5)+cosine-decay
schedule, grad-norm clip 1.0, same CSV splits, same DEPTH_GRIDS/
make_scheduler imported from the repo's train.py). The only difference
from run_train.py: year is also pulled off the loader and passed into the
model, which projects it through its own small MLP into a second
depth-varying channel (see models/fourierbgc.py).

--target_var defaults to CHLA (this repo's original exploratory scope),
but NITRATE and BBP700 both land on the same 200-point depth grid (see
train.py's DEPTH_GRIDS), so the model and this script work unchanged for
all three -- only --target_var and the save dir need to change.

    python scripts/train_fourierbgc.py --seed 0
    python scripts/train_fourierbgc.py --target_var NITRATE --seed 0
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from helpers.dataset import FloatDataset  # noqa: E402
from train import DEPTH_GRIDS, make_scheduler  # noqa: E402
from helpers.seeding import set_seed, seeded_generator  # noqa: E402

from models import FourierBGC  # noqa: E402

DATA_DIR = os.path.join(_REPO_ROOT, "data")


def run_epoch(model, loader, depth_levels, device, optimizer=None, max_grad_norm=1.0):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    losses = []
    with torch.set_grad_enabled(train_mode):
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            profile = torch.stack([temp, psal, doxy], dim=-1).to(device)  # (B, D, 3)
            target = target.unsqueeze(-1).to(device)                      # (B, D, 1)
            day_rad, lat, lon = day_rad.to(device), lat.to(device), lon.to(device)
            year = year.to(device)

            output = model(profile, depth_levels, day_rad, lat, lon, year)
            loss = mse_loss(output, target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            losses.append(loss.item())

    return np.mean(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], default="CHLA")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_dir", default=None,
                    help="default: fourier_learned_projection/results_with_year/"
                         "{target_var}/seed{N}")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: fourierbgc, "
          f"target: {args.target_var}, seed: {args.seed}")

    train_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_train.csv"))
    test_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_test.csv"))
    train_generator = seeded_generator(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               generator=train_generator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = FourierBGC().to(device)
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs)

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_with_year",
        args.target_var, f"seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    best_test_mse = float("inf")
    with open(log_path, "w") as log:
        for ep in range(args.epochs):
            train_mse = run_epoch(model, train_loader, depth_levels, device, optimizer)
            test_mse = run_epoch(model, test_loader, depth_levels, device, optimizer=None)
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
