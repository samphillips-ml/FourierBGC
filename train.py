"""
Trains RawTransformerProbe or RawCNNProbe on PPCon's own dataset (T/S/O only,
no lat/lon/day/year). PPCon's published RMSE is the comparison target, not
something retrained here.

    python train.py --model transformer --target_var NITRATE
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
from models import RawTransformerProbe, RawCNNProbe

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
    elif name == "cnn":
        return RawCNNProbe()
    raise ValueError(f"unknown model {name}")


def run_epoch(model, loader, depth_levels, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    losses = []
    with torch.set_grad_enabled(train_mode):
        for _year, _day_rad, _lat, _lon, temp, psal, doxy, target in loader:
            # raw probe: T/S/O only, geolocation/date dropped on purpose
            profile = torch.stack([temp, psal, doxy], dim=-1).to(device)  # (B, D, 3)
            target = target.unsqueeze(-1).to(device)                      # (B, D, 1)

            output = model(profile, depth_levels)
            loss = mse_loss(output, target)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

    return np.mean(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["transformer", "cnn"], required=True)
    p.add_argument("--target_var", choices=["NITRATE", "CHLA", "BBP700"], required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, model: {args.model}, target: {args.target_var}")

    train_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_train.csv"))
    test_ds = FloatDataset(os.path.join(DATA_DIR, args.target_var, "float_ds_sf_test.csv"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(args.model).to(device)
    depth_levels = DEPTH_GRIDS[args.target_var].to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.results_dir, args.target_var, args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    best_test_mse = float("inf")
    with open(log_path, "w") as log:
        for ep in range(args.epochs):
            train_mse = run_epoch(model, train_loader, depth_levels, device, optimizer)
            test_mse = run_epoch(model, test_loader, depth_levels, device, optimizer=None)

            line = f"epoch {ep+1:4d}  train_mse {train_mse:.5f}  test_mse {test_mse:.5f}"
            print(line)
            log.write(line + "\n")

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    torch.save(model.state_dict(), os.path.join(save_dir, "final.pt"))
    print(f"best test_mse: {best_test_mse:.5f}, saved to {save_dir}/best.pt")


if __name__ == "__main__":
    main()