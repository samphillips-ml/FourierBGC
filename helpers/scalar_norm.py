"""
Z-score constants for the four coordinates, computed over the pooled training
splits of all three variables (n=10,058 profiles). Per-variable statistics
differ by less than 10% of one standard deviation, so one shared set is used.

CNN-RawCoord normalizes all four; FourierBGC and FourierBGC-Broadcast use the
year entry only.
"""
import torch

SCALAR_STATS = {
    "year":    (2016.724976, 2.104907),
    "day_rad": (3.156146, 1.780836),
    "lat":     (38.197697, 3.341911),
    "lon":     (16.424650, 8.562891),
}


def normalize_scalars(lat, lon, day_rad, year):
    """Z-score each scalar. Argument order is (lat, lon, day_rad, year), matching
    the broadcast order in train.py and evaluate.py."""
    lat_mean, lat_std = SCALAR_STATS["lat"]
    lon_mean, lon_std = SCALAR_STATS["lon"]
    day_mean, day_std = SCALAR_STATS["day_rad"]
    year_mean, year_std = SCALAR_STATS["year"]
    return (
        (lat - lat_mean) / lat_std,
        (lon - lon_mean) / lon_std,
        (day_rad - day_mean) / day_std,
        (year - year_mean) / year_std,
    )
