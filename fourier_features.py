"""
Fourier feature encoding of lat, lon, day_of_year for FourierBGC. Pure
sin/cos basis, no learned parameters, no numpy — every value lands in
[-1, 1] by construction, which is why these can be concatenated directly
at the input (unlike the raw-scale broadcast scalars used by cnn_scalar,
which forced an instability fix; see models.py). Year is not encoded here
or anywhere in the FourierBGC pathway: water mass (T/S/O) is the dominant
signal for these targets and inter-annual drift adds nothing principled.

3 harmonics (1x, 2x, 4x) per input, sin+cos each -> 6 values per input,
18 total: [day(6), lat(6), lon(6)].

Lat/lon are normalized to [0, 1] over the Mediterranean's bounding box
before the 2*pi*k*norm angle is taken; day_rad arrives already in
[0, 2*pi] from FloatDataset, so it's used directly.
"""
import torch

LAT_MIN, LAT_SPAN = 31.5, 13.0
LON_MIN, LON_SPAN = 0.57, 38.0
HARMONICS = (1, 2, 4)


def compute_fourier_features(day_rad, lat, lon):
    """day_rad, lat, lon: (B,) tensors. Returns (B, 18) tensor, order
    [day(6), lat(6), lon(6)], each pair (sin, cos) per harmonic 1, 2, 4."""
    lat_norm = (lat - LAT_MIN) / LAT_SPAN
    lon_norm = (lon - LON_MIN) / LON_SPAN

    feats = []
    for k in HARMONICS:
        feats.append(torch.sin(k * day_rad))
        feats.append(torch.cos(k * day_rad))
    for k in HARMONICS:
        feats.append(torch.sin(2 * torch.pi * k * lat_norm))
        feats.append(torch.cos(2 * torch.pi * k * lat_norm))
    for k in HARMONICS:
        feats.append(torch.sin(2 * torch.pi * k * lon_norm))
        feats.append(torch.cos(2 * torch.pi * k * lon_norm))

    return torch.stack(feats, dim=-1)  # (B, 18)
