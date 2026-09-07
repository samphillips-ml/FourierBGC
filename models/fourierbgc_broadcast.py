"""
FourierBGC-Broadcast: the fixed Fourier encoding with no learned projection.

Broadcasts the 18-dimensional Fourier encoding of day-of-year, latitude and
longitude (helpers/fourier_features.py, harmonics 1/2/4) plus the z-scored
year directly along the depth dimension, using no learned parameters at all.
So C = 22: 3 profile channels + 18 Fourier + 1 year.

Comparing this against FourierBGC isolates exactly what the learned projection
contributes beyond the fixed encoding.

It should be noted that year is deliberately left unencoded: 
it is not periodic over 2012-2020 and does not belong in the 
sin/cos family day/lat/lon use.

Broadcasting and concatenation are done by the caller (train.py / evaluate.py).
"""
from models.backbone import Conv1dMed


class FourierBGCBroadcast(Conv1dMed):
    def __init__(self, in_channels=22, dp_rate=0.2):
        super().__init__(in_channels=in_channels, dp_rate=dp_rate)
