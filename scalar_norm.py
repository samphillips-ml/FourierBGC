"""
Z-score constants for the four broadcast scalar channels (lat, lon, day_rad,
year) used by cnn_scalar/transformer_scalar. Computed from the pooled
training splits of NITRATE, CHLA, and BBP700 (n=10,058 profiles); per-variable
stats differ by <10% of one std, so a single shared set is used rather than
one per target_var. T/S/O are untouched, they're out of scope for this fix.

To reproduce: load each data/{VAR}/float_ds_sf_train.csv with FloatDataset,
collect year/day_rad/lat/lon across all profiles, .mean()/.std().
"""
import torch

SCALAR_STATS = {
    "year":    (2016.724976, 2.104907),
    "day_rad": (3.156146, 1.780836),
    "lat":     (38.197697, 3.341911),
    "lon":     (16.424650, 8.562891),
}


def normalize_scalars(lat, lon, day_rad, year):
    """Z-score each scalar with the constants above. Order matches the
    (lat, lon, day_rad, year) broadcast order used in train.py/evaluate.py."""
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
