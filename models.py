"""
Raw probes for the PPCon comparison. Both take T/S/O only, no lat/lon/day/year.
Both return per-depth predictions, shape (B, D, 1), so train.py can swap models
without touching the loss/eval code.
"""
import math

import torch
import torch.nn as nn


class RawTransformerProbe(nn.Module):
    """Lean version: 100,289 params (d_model=64, 2 layers). For the parameter-
    matched comparison later, bump to d_model=128 -> ~397K, just under PPCon's
    412,049 total."""

    def __init__(self, n_in=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.d_model = d_model

    def _sinusoidal_pe(self, depth_values, device):
        # physical depth in meters, normalized to [0,1]. non-uniform grid
        # (5m steps for nitrate, 1m for chla/bbp700) means index-based PE
        # would misrepresent distances, so this takes real depth values.
        depth_norm = depth_values / depth_values.max()
        d = self.d_model
        pe = torch.zeros(len(depth_values), d, device=device)
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(depth_norm.unsqueeze(1) * div)
        pe[:, 1::2] = torch.cos(depth_norm.unsqueeze(1) * div)
        return pe

    def forward(self, profile, depth_levels):
        # profile: (B, D, 3), depth_levels: (D,) physical depth in meters
        x = self.input_proj(profile)
        x = x + self._sinusoidal_pe(depth_levels, profile.device).unsqueeze(0)
        x = self.transformer(x)
        return self.head(x)  # (B, D, 1)


class RawCNNProbe(nn.Module):
    """PPCon's own Conv1dMed backbone, in_channels 7 -> 3 (drops the four
    MLP-expanded day/year/lat/lon channels, keeps only temp/psal/doxy).
    Same kernels, strides, padding, dropout placement, and the SELU-before-BN
    ordering as their original, adapted from github.com/gpietrop/PPCon (MIT).
    This is the same-architecture-family control: same conv stack PPCon used,
    minus the geolocation pathway, so a win/loss here isolates the input-set
    effect from the architecture effect.

    253,249 params at in_channels=7 (PPCon's number), 252,737 at in_channels=3
    (this one). Dropping 4 of 7 input channels costs only 512 params, all in
    conv1, since channel count barely touches a conv layer's parameter count
    compared to kernel size and channel depth elsewhere in the stack.
    """

    def __init__(self, in_channels=3, dp_rate=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=2, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.af1 = nn.SELU()
        self.do1 = nn.Dropout(p=dp_rate)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.af2 = nn.SELU()
        self.do2 = nn.Dropout(p=dp_rate)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.af3 = nn.SELU()
        self.do3 = nn.Dropout(p=dp_rate)

        self.conv12 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=2)
        self.bn12 = nn.BatchNorm1d(128)
        self.af12 = nn.SELU()
        self.do12 = nn.Dropout(p=dp_rate)

        self.deconv13 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=2)
        self.bn13 = nn.BatchNorm1d(128)
        self.af13 = nn.SELU()
        self.do13 = nn.Dropout(p=dp_rate)

        self.conv14 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm1d(128)
        self.af14 = nn.SELU()
        self.do14 = nn.Dropout(p=dp_rate)

        self.deconv15 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=1)
        self.bn15 = nn.BatchNorm1d(64)
        self.af15 = nn.SELU()
        self.do15 = nn.Dropout(p=dp_rate)

        self.conv16 = nn.Conv1d(64, 32, kernel_size=2, stride=2, padding=1)
        self.bn16 = nn.BatchNorm1d(32)
        self.af16 = nn.SELU()
        self.do16 = nn.Dropout(p=dp_rate)

        self.conv17 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, profile, depth_levels=None):
        # profile: (B, D, 3) -> conv1d wants (B, C, D). depth_levels unused,
        # kept in the signature so train.py can call both probes identically.
        x = profile.transpose(1, 2)
        x = self.bn1(self.af1(self.conv1(x))); x = self.do1(x)
        x = self.bn2(self.af2(self.conv2(x))); x = self.do2(x)
        x = self.bn3(self.af3(self.conv3(x))); x = self.do3(x)
        x = self.bn12(self.af12(self.conv12(x))); x = self.do12(x)
        x = self.bn13(self.af13(self.deconv13(x))); x = self.do13(x)
        x = self.bn14(self.af14(self.conv14(x))); x = self.do14(x)
        x = self.bn15(self.af15(self.deconv15(x))); x = self.do15(x)
        x = self.bn16(self.af16(self.conv16(x))); x = self.do16(x)
        x = self.conv17(x)
        return x.transpose(1, 2)  # back to (B, D, 1)

# PPcon's direct model copied from https://github.com/gpietrop/ppcon/blob/main/ppcon/train/conv1med_dp.py

in_channels = 7
out_channels = 1
class Conv1dMed(nn.Module):
    def __init__(self, dp_rate):
        """
        Initializes the Conv1dMed model.

        :param dp_rate: float
            The dropout rate to be applied after each layer.
        """
        super(Conv1dMed, self).__init__()
        self.dp_rate = dp_rate

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=2, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.af1 = nn.SELU()  # nn.SELU()
        self.do1 = nn.Dropout(p=self.dp_rate)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.af2 = nn.SELU()  # nn.SELU()
        self.do2 = nn.Dropout(p=self.dp_rate)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.af3 = nn.SELU()  # nn.SELU()
        self.do3 = nn.Dropout(p=self.dp_rate)

        self.conv12 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=2)
        self.bn12 = nn.BatchNorm1d(128)
        self.af12 = nn.SELU()  # nn.SELU()
        self.do12 = nn.Dropout(p=self.dp_rate)

        self.deconv13 = nn.ConvTranspose1d(128, 128, kernel_size=2, stride=2, padding=2)
        self.bn13 = nn.BatchNorm1d(128)
        self.af13 = nn.SELU()  # nn.SELU()
        self.do13 = nn.Dropout(p=self.dp_rate)

        self.conv14 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm1d(128)
        self.af14 = nn.SELU()  # nn.SELU()
        self.do14 = nn.Dropout(p=self.dp_rate)

        self.deconv15 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=1)
        self.bn15 = nn.BatchNorm1d(64)
        self.af15 = nn.SELU()  # nn.SELU()
        self.do15 = nn.Dropout(p=self.dp_rate)

        self.conv16 = nn.Conv1d(64, 32, kernel_size=2, stride=2, padding=1)
        self.bn16 = nn.BatchNorm1d(32)
        self.af16 = nn.SELU()  # nn.SELU()
        self.do16 = nn.Dropout(p=self.dp_rate)

        self.conv17 = nn.Conv1d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        :param x: torch.Tensor
            The input tensor of shape (batch_size, in_channels, sequence_length).

        :return: torch.Tensor
            The output tensor after passing through the network.
        """
        x = self.bn1(self.af1(self.conv1(x)))
        x = self.do1(x)
        x = self.bn2(self.af2(self.conv2(x)))
        x = self.do2(x)
        x = self.bn3(self.af3(self.conv3(x)))
        x = self.do3(x)
        x = self.bn12(self.af12(self.conv12(x)))
        x = self.do12(x)
        x = self.bn13(self.af13(self.deconv13(x)))
        x = self.do13(x)
        x = self.bn14(self.af14(self.conv14(x)))
        x = self.do14(x)
        x = self.bn15(self.af15(self.deconv15(x)))
        x = self.do15(x)
        x = self.bn16(self.af16(self.conv16(x)))
        x = self.do16(x)
        x = self.conv17(x)

        return x
## PPcon's MLP models
import torch.nn as nn


class MLPDay(nn.Module):
    def __init__(self):
        super(MLPDay, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 80),
            nn.SELU(),
            nn.Linear(80, 140),
            nn.SELU(),
            nn.Linear(140, 200),
            nn.SELU()
        )

    def forward(self, x):
        # x = x.view(-1, self.input_size)
        return self.network(x)


class MLPYear(nn.Module):
    def __init__(self):
        super(MLPYear, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 80),
            nn.SELU(),
            nn.Linear(80, 140),
            nn.SELU(),
            nn.Linear(140, 200),
            nn.SELU()
        )

    def forward(self, x):
        # x = x.view(-1, self.input_size)
        return self.network(x)


class MLPLat(nn.Module):
    def __init__(self):
        super(MLPLat, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 80),
            nn.SELU(),
            nn.Linear(80, 140),
            nn.SELU(),
            nn.Linear(140, 200),
            nn.SELU()
        )

    def forward(self, x):
        return self.network(x)


class MLPLon(nn.Module):
    def __init__(self):
        super(MLPLon, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 80),
            nn.SELU(),
            nn.Linear(80, 140),
            nn.SELU(),
            nn.Linear(140, 200),
            nn.SELU()
        )

    def forward(self, x):
        return self.network(x)
def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    t = RawTransformerProbe()
    c = RawCNNProbe()
    print(f"RawTransformerProbe: {count_params(t):,} params")
    print(f"RawCNNProbe:         {count_params(c):,} params")
    print(f"PPCon total:         412,049 params  (158,800 in 4 scalar MLPs, 253,249 in conv stack)")