"""
Trains RawTransformerProbe, RawTransformerScalarProbe, RawCNNProbe, or
RawCNNScalarProbe on PPCon's own dataset. transformer/cnn use T/S/O only, no
lat/lon/day/year. transformer_scalar/cnn_scalar add those four scalars back
in as constant-valued depth channels (broadcast here, no learned encoding).
PPCon's published RMSE is the comparison target, not something retrained
here.

    python train.py --model transformer --target_var NITRATE
    python train.py --model transformer_scalar --target_var NITRATE
    python train.py --model cnn --target_var CHLA --epochs 100
"""
import argparse
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from dataset import FloatDataset
from models import (RawTransformerProbe, RawTransformerScalarProbe, RawCNNProbe,
                     RawCNNScalarProbe)
from scalar_norm import normalize_scalars

# PPCon's own depth grids (dict.py): nitrate 0-1000m @ 5m, chla/bbp700 0-200m @ 1m.
# both land at 200 points but they're physically different distances.
DEPTH_GRIDS = {
    "NITRATE": torch.arange(0, 1000, 5).float(),
    "CHLA":    torch.arange(0, 200, 1).float(),
    "BBP700":  torch.arange(0, 200, 1).float(),
}

DATA_DIR = "data"


def make_model(name):
    if name == "transformer":
        return RawTransformerProbe()
    elif name == "transformer_scalar":
        return RawTransformerScalarProbe()
    elif name == "cnn":
        return RawCNNProbe()
    elif name == "cnn_scalar":
        return RawCNNScalarProbe()
    raise ValueError(f"unknown model {name}")


def run_epoch(model, loader, depth_levels, device, optimizer=None, max_grad_norm=1.0,
              use_scalars=False):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    n_depth = depth_levels.shape[0]
    losses = []
    with torch.set_grad_enabled(train_mode):
        for year, day_rad, lat, lon, temp, psal, doxy, target in loader:
            # raw probe: T/S/O only, geolocation/date dropped on purpose
            profile = torch.stack([temp, psal, doxy], dim=-1).to(device)  # (B, D, 3)
            target = target.unsqueeze(-1).to(device)                      # (B, D, 1)

            if use_scalars:
                # broadcast each scalar to a constant-valued depth channel,
                # no learned encoding (that's the point of this ablation).
                # z-scored first, raw lat/lon/day_rad/year are on wildly
                # different scales from each other and from T/S/O.
                b = profile.shape[0]
                lat, lon, day_rad, year = normalize_scalars(lat, lon, day_rad, year)
                scalars = torch.stack([lat, lon, day_rad, year], dim=-1).to(device)  # (B, 4)
                scalars = scalars.view(b, 1, 4).expand(b, n_depth, 4)
                profile = torch.cat([profile, scalars], dim=-1)  # (B, D, 7)

            output = model(profile, depth_levels)
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
    p.add_argument("--model", choices=["transformer", "transformer_scalar", "cnn", "cnn_scalar"],
                   required=True)
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    #elif torch.backends.mps.is_available():
     #   device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}, model: {args.model}, target: {args.target_var}")

    train_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_train.csv"))
    test_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_test.csv"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(args.model).to(device)
    use_scalars = args.model in ("transformer_scalar", "cnn_scalar")
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.epochs)

    save_dir = os.path.join(args.results_dir, args.target_var, args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    best_test_mse = float("inf")
    with open(log_path, "w") as log:
        for ep in range(args.epochs):
            train_mse = run_epoch(model, train_loader, depth_levels, device, optimizer,
                                   use_scalars=use_scalars)
            test_mse = run_epoch(model, test_loader, depth_levels, device, optimizer=None,
                                  use_scalars=use_scalars)
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