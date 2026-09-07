"""
FourierBGC: the paper's headline model.

The same fixed 18-dimensional Fourier encoding as FourierBGC-Broadcast, but
projected through a small learned network into a single depth-varying channel,
with year z-scored and projected separately through its own smaller network
into a second channel. C = 5: 3 profile channels + Fourier projection + year
projection.

    fourier_proj: 18 -> 32 -> 64 -> 200  (15,720 params)
    year_proj:     1 -> 16 -> 200        ( 3,432 params)

19,152 parameters in total against the 158,800 PPCon spends on four separately
parameterized encoders -- an 88 % reduction -- occupying the same position in
the network as PPCon's four encoder outputs.

The three coordinates share one projection rather than receiving separate
networks. That is the simplest configuration that isolates the effect of the
fixed encoding against CNN-MLPCoord's learned per-coordinate encoders without
introducing further un-ablated architectural choices. Whether splitting the
projection per coordinate changes the result is left to future work.
"""
import torch
import torch.nn as nn

from models.backbone import Conv1dMed
from helpers.fourier_features import compute_fourier_features
from helpers.scalar_norm import SCALAR_STATS

N_FOURIER_FEATURES = 18
N_DEPTH = 200  # PPCon's depth grids all land on 200 points


class FourierBGC(Conv1dMed):
    def __init__(self, in_channels=5, dp_rate=0.2, fourier_hidden=(32, 64), year_hidden=16):
        super().__init__(in_channels=in_channels, dp_rate=dp_rate)
        h1, h2 = fourier_hidden
        self.fourier_proj = nn.Sequential(
            nn.Linear(N_FOURIER_FEATURES, h1),
            nn.SELU(),
            nn.Linear(h1, h2),
            nn.SELU(),
            nn.Linear(h2, N_DEPTH),
            nn.SELU(),
        )
        self.year_proj = nn.Sequential(
            nn.Linear(1, year_hidden),
            nn.SELU(),
            nn.Linear(year_hidden, N_DEPTH),
            nn.SELU(),
        )

    def forward(self, profile, depth_levels, day_rad, year, lat, lon):
        # Argument order follows the paper's Eq. (2), c = (d, t, phi, lambda),
        # and matches CNNMLPCoord.forward. Reordering forward() arguments does
        # not affect checkpoints: state_dict keys are layer attribute paths.
        fourier = compute_fourier_features(day_rad, lat, lon)        # (B, 18)
        fourier_channel = self.fourier_proj(fourier).unsqueeze(-1)   # (B, 200, 1)

        year_mean, year_std = SCALAR_STATS["year"]
        year_z = ((year.float() - year_mean) / year_std).unsqueeze(-1)  # (B, 1)
        year_channel = self.year_proj(year_z).unsqueeze(-1)             # (B, 200, 1)

        x = torch.cat([profile, fourier_channel, year_channel], dim=-1)  # (B, D, 5)
        return super().forward(x, depth_levels)

    def encoder_params(self):
        return (sum(p.numel() for p in self.fourier_proj.parameters())
                + sum(p.numel() for p in self.year_proj.parameters()))
