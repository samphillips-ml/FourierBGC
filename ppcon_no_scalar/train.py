"""
PPConNoScalar's training function. Near-verbatim copy of
ppcon_baseline/ppcon/train/train.py's train_model, with exactly five
structural changes from the original, all driven by removing the four
scalar MLP inputs (day_rad, year, lat, lon):

  1. no MLPDay/MLPYear/MLPLat/MLPLon instantiation
  2. no MLP .to(device) calls
  3. no MLP params in the Adadelta optimizer
  4. no MLP forward passes (in either the training or validation loop)
  5. torch.cat takes (temp, psal, doxy) -- 3 channels, not 7

Consequences of removing the MLPs: only model_conv_{epoch}.pt is saved per
snapshot (not five state dicts), and the peak_difference term is dropped
(attention_max is always 0 in how PPCon was actually run, so this isn't a
behavioral change). Everything else -- Adadelta, the three-term loss (MSE +
L2 + smoothness), EarlyStopping instantiation, snaperiod logic, loss file
writing, print formats -- is preserved exactly from train_model.

Conv1dMed's in_channels is a module-level constant in conv1med_dp.py (not a
constructor arg), so it's monkeypatched to 3 only for the duration of model
construction, then restored, so this process can still construct PPCon's
real 7-channel Conv1dMed elsewhere (e.g. via ppcon_eval.py) without
cross-talk.
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
from IPython import display

import torch
from torch.optim import Adadelta
from torch.nn.functional import mse_loss

_PPCON_BASELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppcon_baseline")
if _PPCON_BASELINE_DIR not in sys.path:
    sys.path.insert(0, _PPCON_BASELINE_DIR)

from ppcon.utils.pytorchtools import EarlyStopping  # noqa: E402  # type: ignore[import]
from ppcon.train import conv1med_dp  # noqa: E402  # type: ignore[import]
from ppcon.train.conv1med_dp import Conv1dMed  # noqa: E402  # type: ignore[import]


@contextmanager
def _conv1dmed_in_channels(n):
    orig = conv1med_dp.in_channels
    conv1med_dp.in_channels = n
    try:
        yield
    finally:
        conv1med_dp.in_channels = orig


def train_ppcon_no_scalar(train_loader, val_loader, epoch, lr, dp_rate, lambda_l2_reg, alpha_smooth_reg, snaperiod,
                           device, dir, flag_early_stopping=False, verbose=False):
    """
    Trains Conv1dMed on only the T/S/O channels (no scalar MLP inputs).

    :param train_loader: DataLoader
        The DataLoader providing the training data.
    :param val_loader: DataLoader
        The DataLoader providing the validation data.
    :param epoch: int
        The number of epochs to train the model.
    :param lr: float
        The learning rate for the optimizer.
    :param dp_rate: float
        The dropout rate used in the convolutional model.
    :param lambda_l2_reg: float
        The L2 regularization coefficient applied to the model weights.
    :param alpha_smooth_reg: float
        The smoothing regularization coefficient to enforce smoothness in the output.
    :param snaperiod: int
        The number of epochs between saving model snapshots.
    :param device: torch.device
        The device (CPU or GPU) on which the model will be trained and evaluated.
    :param dir: str
        The directory where the model, logs, and checkpoints will be saved.
    :param flag_early_stopping: bool, optional
        If True, enables early stopping based on training loss. Default is False.
    :param verbose: bool, optional
        If True, prints detailed information during training. Default is False.

    :return: None
    """

    save_dir = dir + "/model/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with _conv1dmed_in_channels(3):
        model_conv = Conv1dMed(dp_rate=dp_rate)

    # Moving model to GPU when available
    model_conv.to(device)

    params = list(model_conv.parameters())
    optimizer = Adadelta(params=params, lr=lr)

    f, f_test = open(dir + "/train_loss.txt", "w+"), open(dir + "/test_loss.txt", "w+")
    f_mse_train, f_mse_test = open(dir + "/mse_train_loss.txt", "w+"), open(dir + "/mse_test_loss.txt", "w+")

    # initialize the early_stopping object
    path_checkpoint = save_dir + 'checkpoint.pt'
    early_stopping = EarlyStopping(patience=5, verbose=True, path=path_checkpoint, )

    for ep in range(epoch + 1):
        loss_train = []
        loss_test = []
        mse_train = []
        mse_test = []
        # Model in training mode
        model_conv.train()

        for training_year, training_day, training_lat, training_lon, training_temp, training_psal, training_doxy, training_output in train_loader:

            # Moving tensors to GPU when available
            training_temp = training_temp.to(device)
            training_psal = training_psal.to(device)
            training_doxy = training_doxy.to(device)
            training_output = training_output.to(device)

            training_temp = torch.transpose(training_temp.unsqueeze(0), 0, 1)
            training_psal = torch.transpose(training_psal.unsqueeze(0), 0, 1)
            training_doxy = torch.transpose(training_doxy.unsqueeze(0), 0, 1)
            training_output = torch.transpose(training_output.unsqueeze(0), 0, 1)

            training_x = torch.cat((training_temp, training_psal, training_doxy), 1)

            output = model_conv(training_x.float())

            mse = mse_loss(training_output, output)

            l2_norm = sum(p.pow(2.0).sum() for p in model_conv.parameters())
            l2_reg = lambda_l2_reg * l2_norm

            smoothness = 0
            for index in range(output.shape[0]):
                batch_tens = output[index, 0, :]
                batch_tens_smoothness = sum(torch.abs(batch_tens[i] - batch_tens[i - 1]) for i in range(1, batch_tens.shape[0]))
                smoothness += batch_tens_smoothness
            smoothness = alpha_smooth_reg * smoothness

            loss_conv = mse + l2_reg + smoothness

            loss_train.append(loss_conv.item())
            mse_train.append(mse.item())

            if verbose:
                print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_conv.item():.12f}")
                display.clear_output(wait=True)

            optimizer.zero_grad()
            loss_conv.backward()
            optimizer.step()

        avg_train_loss = np.average(loss_train)
        avg_train_mse = np.average(mse_train)

        print(f"[==== EPOCH]: {ep + 1}, [AVERAGE LOSS]: {avg_train_loss:.5f}")
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {avg_train_loss:.5f} \n")
        f_mse_train.write(f"[EPOCH]: {ep + 1}, [LOSS]: {avg_train_mse:.4f} \n")

        # early_stopping needs the training loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        if flag_early_stopping:
            early_stopping(avg_train_loss, model_conv)
            # stop if validation loss doesn't improve after a given patience
            if early_stopping.early_stop:
               print("Early stopping")
               break

        # Saving model and testing
        if ep % snaperiod == 0 or ep == epoch:

            # Saving model at epoch ep
            torch.save(model_conv.state_dict(), save_dir + "/model_conv_" + str(ep) + ".pt")

            # Model in validation mode
            model_conv.eval()

            with torch.no_grad():
                for testing_year, testing_day, testing_lat, testing_lon, testing_temp, testing_psal, testing_doxy, testing_output in val_loader:

                    # Moving tensors to GPU when available
                    testing_temp = testing_temp.to(device)
                    testing_psal = testing_psal.to(device)
                    testing_doxy = testing_doxy.to(device)
                    testing_output = testing_output.to(device)

                    testing_temp = torch.transpose(testing_temp.unsqueeze(0), 0, 1)
                    testing_psal = torch.transpose(testing_psal.unsqueeze(0), 0, 1)
                    testing_doxy = torch.transpose(testing_doxy.unsqueeze(0), 0, 1)
                    testing_output = torch.transpose(testing_output.unsqueeze(0), 0, 1)

                    testing_x = torch.cat((testing_temp, testing_psal, testing_doxy), 1)

                    output_test = model_conv(testing_x.float())

                    mse = mse_loss(testing_output, output_test)

                    l2_norm = sum(p.pow(2.0).sum() for p in model_conv.parameters())
                    l2_reg = lambda_l2_reg * l2_norm

                    smoothness = 0
                    for index in range(output_test.shape[0]):
                        batch_tens = output_test[index, 0, :]
                        batch_tens_smoothness = sum(batch_tens[i] - batch_tens[i-1] for i in range(1, batch_tens.shape[0]))
                        smoothness += batch_tens_smoothness
                    smoothness = alpha_smooth_reg * smoothness

                    loss_conv = mse + l2_reg + smoothness

                    loss_test.append(loss_conv)
                    mse_test.append(mse)

                    if verbose:
                        print(f"-----[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_conv.item():.12f}")
                        display.clear_output(wait=True)

            avg_test_loss = np.average([loss_.cpu() for loss_ in loss_test])
            avg_test_mse = np.average([loss_.cpu() for loss_ in mse_test])

            print(f"[==== EPOCH]: {ep + 1}, [AVERAGE TEST LOSS]: {avg_test_loss:.5f}")
            f_test.write(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {avg_test_loss:.5f} \n")
            f_mse_test.write(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {avg_test_mse:.4f} \n")

    f.close()
    f_test.close()
    f_mse_train.close()
    f_mse_train.close()
