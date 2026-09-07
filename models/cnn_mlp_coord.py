"""
CNN-MLPCoord: PPCon's four learned per-coordinate encoders under the common
training recipe.

The encoder is identical to PPcon, and we apply the common training recipe. 
We do not normalize the inputs, just like PPcon.
"""
import torch
import torch.nn as nn

from models.backbone import Conv1dMed


class CNNMLPCoord(Conv1dMed):
    def __init__(self, in_channels=7, dp_rate=0.2, normalize=False):
        super().__init__(in_channels=in_channels, dp_rate=dp_rate)
        self.normalize = normalize

        def _ppcon_mlp():
            return nn.Sequential(
                nn.Linear(1, 80),
                nn.SELU(),
                nn.Linear(80, 140),
                nn.SELU(),
                nn.Linear(140, 200),
                nn.SELU(),
            )

        # named to match PPCon's own four encoders (MLPDay/MLPYear/MLPLat/MLPLon)
        self.mlp_day = _ppcon_mlp()
        self.mlp_year = _ppcon_mlp()
        self.mlp_lat = _ppcon_mlp()
        self.mlp_lon = _ppcon_mlp()

    def forward(self, profile, depth_levels, day_rad, year, lat, lon):
        # Argument order follows the paper's Eq. (2), c = (d, t, phi, lambda).
        if self.normalize:
            from helpers.scalar_norm import normalize_scalars
            lat, lon, day_rad, year = normalize_scalars(lat, lon, day_rad, year)

        chans = []
        for mlp, s in ((self.mlp_day, day_rad), (self.mlp_year, year),
                       (self.mlp_lat, lat), (self.mlp_lon, lon)):
            chans.append(mlp(s.float().unsqueeze(-1)).unsqueeze(-1))  # (B, 200, 1)

        x = torch.cat([profile] + chans, dim=-1)  # (B, D, 7)
        return super().forward(x, depth_levels)

    def encoder_params(self):
        return sum(p.numel() for m in (self.mlp_day, self.mlp_year,
                                       self.mlp_lat, self.mlp_lon)
                   for p in m.parameters())
