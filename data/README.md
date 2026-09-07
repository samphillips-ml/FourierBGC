Some notes on the copied dataset from PPcon:

1. Profiles are columns, not rows
2. Each profile is 8 rows: year, day-of-year (in radians), lat, lon, temp, psal, doxy, target.
3. Those profile features are stored as *stringified lists* that *from_string_to_tensor* parses.
4. bbp700 targets are stored ×1000 — divide to get m⁻¹. Nitrate and chl-a are as-is.

The variable folders are organized as such:

1. `VARIABLE_NAME/float_ds_sf_removed.csv`
    - Profiles dropped by PPCon's quality control. Nothing here reads them.
2. `VARIABLE_NAME/float_ds_sf_test.csv`
    - The test split, used for every number in the paper.
3. `VARIABLE_NAME/float_ds_sf_train.csv`
    - The training split.
