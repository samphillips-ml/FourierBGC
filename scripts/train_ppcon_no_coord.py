"""
I mirrored third_party/ppcon/ppcon/run_model.py's run_training/CLI.

The differences I made are: 
train_ppcon_no_scalar() instead of train_model()
skips plot_profiles() at the end (plot_profiles imports MLPDay/Conv1dMed 
from ppcon.train.train and is built around the 7-channel scalar+T/S/O concatenation, so it isn't
compatible with a 3-channel-only run; evaluate.py is the actual eval path
used for this ablation).

Results are saved under RESULTS_DIR = ~/ppcon_results_no_scalar/<VARIABLE>/,
mirroring PPCon's own ~/ppcon_results/<VARIABLE>/ convention (config.py),
just under a different home-dir folder so it never collides with PPCon's
real results.

This file is the only one in scripts/ just to make sure its clear that its not a helper,
but also not part of the common train.py at the root
"""
import os
import random
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PPCON_BASELINE_DIR = os.path.join(_REPO_ROOT, "third_party/ppcon")
for _p in (_REPO_ROOT, _PPCON_BASELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ppcon.config import DATASET_DIR  # noqa: E402  # type: ignore[import]
from ppcon.utils.dataset import FloatDataset  # noqa: E402  # type: ignore[import]
from ppcon.utils.utils_train import save_ds_info  # noqa: E402  # type: ignore[import]

from helpers.ppcon_no_coord_train import train_ppcon_no_scalar  # noqa: E402
from helpers.seeding import set_seed, seeded_generator  # noqa: E402

# Setting the computation device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")
random.seed(123)

RESULTS_DIR = os.path.join(str(Path.home()), 'ppcon_results_no_scalar')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_training(
        variable="NITRATE",
        batch_size=32,
        epochs=0,
        lr=1,
        snaperiod=25,
        dropout_rate=0.2,
        lambda_l2_reg=0.001,
        alpha_smooth_reg=0.001,
        flag_early_stopping=False,
        seed=None,
        save_dir_override=None,
):
    """
    Run the training process for PPConNoScalar with the specified parameters.

    :param variable: str, optional
        The target variable to predict (e.g., "NITRATE"). Default is "NITRATE".
    :param batch_size: int, optional
        The number of samples per batch during training. Default is 32.
    :param epochs: int, optional
        The number of training epochs. Default is 0.
    :param lr: float, optional
        The learning rate for the optimizer. Default is 1.
    :param snaperiod: int, optional
        The period (in epochs) at which to save model snapshots. Default is 25.
    :param dropout_rate: float, optional
        The dropout rate to use in the model for regularization. Default is 0.2.
    :param lambda_l2_reg: float, optional
        The L2 regularization coefficient. Default is 0.001.
    :param alpha_smooth_reg: float, optional
        The smoothing regularization coefficient. Default is 0.001.
    :param flag_early_stopping: bool, optional
        If True, apply early stopping during training. Default is False.

    :return: None
    """

    # ===== Printing information about the run
    print(f"The variable predicted is {variable}\n"
          f"The total number of epochs that will be performed is {epochs}\n"
          f"seed: {seed}")

    if seed is not None:
        set_seed(seed)

    train_dataset = FloatDataset(os.path.join(DATASET_DIR, variable, 'float_ds_sf_train.csv'))
    val_dataset = FloatDataset(os.path.join(DATASET_DIR, variable, 'float_ds_sf_test.csv'))

    train_generator = seeded_generator(seed) if seed is not None else None
    val_generator = seeded_generator(seed) if seed is not None else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=val_generator)

    save_dir = save_dir_override if save_dir_override else (RESULTS_DIR + "/" + variable)
    os.makedirs(save_dir, exist_ok=True)
    print(f"saving results in {save_dir}")

    # ===== Saving models hyperparameters
    save_ds_info(training_folder="default", batch_size=batch_size, epochs=epochs, lr=lr,
                 dp_rate=dropout_rate, lambda_l2_reg=lambda_l2_reg, save_dir=save_dir,
                 alpha_smooth_reg=alpha_smooth_reg)

    # ===== train the model
    train_ppcon_no_scalar(train_loader, val_loader, epoch=epochs, lr=lr, dp_rate=dropout_rate,
                           lambda_l2_reg=lambda_l2_reg, alpha_smooth_reg=alpha_smooth_reg, snaperiod=snaperiod,
                           dir=save_dir, device=device, flag_early_stopping=flag_early_stopping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str, default="NITRATE", choices=["NITRATE", "CHLA", "BBP700"])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--snaperiod', type=int, default=25)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--lambda_l2_reg', type=float, default=0.001)
    parser.add_argument('--alpha_smooth_reg', type=float, default=0.001)
    parser.add_argument('--flag_early_stopping', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=None,
                         help="optional seed for reproducibility: seeds python/numpy/torch RNG "
                              "and the train/val DataLoader shuffle order. Default None preserves "
                              "the old unseeded behavior (module-level random.seed(123) above only "
                              "ever seeded python's random module, never torch, so it never actually "
                              "controlled weight init or batch order).")
    parser.add_argument('--save_dir', type=str, default=None,
                         help="override the computed save dir (RESULTS_DIR/variable); used to route "
                              "seeded runs to their own directory tree")

    args = parser.parse_args()

    run_training(
        variable=args.variable,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        snaperiod=args.snaperiod,
        dropout_rate=args.dropout_rate,
        lambda_l2_reg=args.lambda_l2_reg,
        alpha_smooth_reg=args.alpha_smooth_reg,
        flag_early_stopping=args.flag_early_stopping,
        seed=args.seed,
        save_dir_override=args.save_dir,
    )
