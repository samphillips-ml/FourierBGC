# PPCon baseline provenance

Vendored from https://github.com/gpietrop/ppcon
Commit: dcc5ed92c192a3e71d42c80de08e6cd33f85a290
License: MIT (see LICENSE in this directory)

Unmodified snapshot except for one patch, below.

## How PPCon's numbers in the manuscript were produced

PPCon is **not retrained**. The external baseline is the authors' released
checkpoint, shipped in this snapshot at ppcon/results/{VAR}/{date}/model/ and
copied to results/ppcon/ for convenience. Epochs 100 / 150 / 125 for
NITRATE / CHLA / BBP700, per their dict.py.

Two evaluation paths were used, which is why the table captions differ:

  - Table 4 (pooled RMSE): this repository's evaluate.py, run once. It is
    deterministic (shuffle=False, batch_size=1), so there is no seed spread.
    Verified 7 Sep 2026: reproduces 0.5234 / 0.0802 / 2.153 exactly.

  - Tables 5, 6, 7 (region and season): PPCon's own evaluation code, median
    of 11 independent re-runs. Their code shuffles unseeded and has an
    off-by-one in its reconstruction routine, so repeated evaluations of the
    same checkpoint differ; the median guards against outlying runs. These
    values do NOT reproduce from this repository's evaluate.py, which is
    expected -- different code, different bucketing behaviour.

## Not used by the manuscript

An earlier phase of this project did retrain PPCon locally. That run is
preserved at archive/home--ppcon_results/ and produced a different checkpoint
(SHA-1 9753bbaa... against the released bad47f5c...). It backs nothing in the
paper. PPCon's config.py writes training output to ~/ppcon_results, never to
ppcon/results/, which is what keeps the two distinguishable.

## Patch applied

ppcon/utils/pytorchtools.py: np.Inf -> np.inf, for NumPy 2.0 compatibility.
One-character change, no behavioural effect.
