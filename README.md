# FourierBGC

Code and checkpoints for *Fourier Feature Encoding for Spatiotemporal Inputs in CNN-Based
Reconstruction of Biogeochemical Argo Profiles* (Phillips & Li).

A controlled ablation over how spatiotemporal coordinates are represented for reconstructing
biogeochemical profiles from BGC-Argo floats. PPCon's convolutional backbone is held fixed
across seven models that differ only in the coordinate encoding and, for two of them, the
training procedure.

## Layout

    data/           PPCon's published train/test splits, inherited unmodified
    models/         one module per model in the ablation spine, plus the shared backbone
    helpers/        dataset loader, Fourier basis, z-score constants, seeding, PPCon adapter
    scripts/        the two training entry points that are not train.py
    results/        every checkpoint behind every table (see results/MANIFEST.md)
    third_party/    the vendored PPCon baseline, unmodified
    visualizations/ figure-generating code and the figures themselves
    archive/        full run trees and superseded runs; gitignored, not needed
    train.py        common-recipe training
    evaluate.py     evaluation, reproducing PPCon's own RMSE methodology

## Model name map

The manuscript and the code use different names. `models/__init__.py` holds the mapping;
this is it in prose. Legacy flags still resolve, so older commands keep working.

| Paper | Registry key | Legacy flag | Class | C | Encoder params |
|---|---|---|---|---|---|
| PPCon | `ppcon` | — | vendored | 7 | 158,800 |
| PPCon-NoCoord | `ppcon_no_coord` | `ppcon_no_scalar` | loader, not a class | 3 | 0 |
| CNN-NoCoord | `cnn_no_coord` | `cnn` | `CNNNoCoord` | 3 | 0 |
| CNN-RawCoord | `cnn_raw_coord` | `cnn_scalar` | `CNNRawCoord` | 3 | 0 |
| CNN-MLPCoord | `cnn_mlp_coord` | `cnn_mlpcoord` | `CNNMLPCoord` | 7 | 158,800 |
| FourierBGC-Broadcast | `fourierbgc_broadcast` | `fourierbgc_with_year` | `FourierBGCBroadcast` | 22 | 0 |
| FourierBGC | `fourierbgc` | `fourierbgc_learned` | `FourierBGC` | 5 | 19,152 |

`C` is the backbone input channel count.

## Data

`data/{NITRATE,CHLA,BBP700}/float_ds_sf_{train,test}.csv`, inherited unmodified from PPCon's
published splits, which in turn apply the quality control of Amadio et al. (2023). Not
reprocessed here. Every model reads from this one location.

## Training

The common recipe: Adam at 1e-3, five-epoch linear warmup then cosine decay, gradient
clipping at max norm 1.0, 100 epochs, batch size 32, five seeds (0–4).

    python train.py --model cnn_no_coord         --target_var NITRATE --seed 0
    python train.py --model cnn_raw_coord        --target_var CHLA    --seed 0
    python train.py --model cnn_mlp_coord        --target_var BBP700  --seed 0
    python train.py --model fourierbgc_broadcast --target_var NITRATE --seed 0

FourierBGC computes its Fourier projection inside the model, so it has its own entry point:

    python scripts/train_fourierbgc.py --target_var NITRATE --epochs 100 --seed 0

PPCon-NoCoord uses PPCon's own recipe (Adadelta at lr 1.0, three-term loss, no clipping or
schedule) and PPCon's per-variable epoch counts, 100 / 150 / 125:

    python scripts/train_ppcon_no_coord.py --variable NITRATE --epochs 100 --seed 0

PPCon itself is never retrained; the authors' released checkpoint is evaluated directly.

## Evaluation

Reproduces PPCon's own RMSE methodology: per-profile RMSE, then averaged within region and
season buckets, weighted by count. Region and season definitions follow PPCon's released
evaluation code, not their Table 4, which disagrees with it on the Ionian Sea boundary.

    python evaluate.py --model fourierbgc --target_var CHLA \
        --checkpoint results/fourierbgc/seed0/CHLA/best.pt

    python evaluate.py --model ppcon --target_var CHLA \
        --checkpoint_dir results/ppcon/CHLA/model --epoch 150

    python evaluate.py --model ppcon_no_coord --target_var CHLA \
        --checkpoint_dir results/ppcon_no_coord/seed0/CHLA/model --epoch 150

**Caveat:** `evaluate.py` prints RMSE with `%.4f`. bbp700 values are around 2e-4, so that is a
single significant figure and every printed bbp700 value collapses to `0.0002`. This has
caused two real errors. Read bbp700 at full precision.

## Reproducing the manuscript

`results/` holds all 105 checkpoints, one per model × variable × seed. `results/MANIFEST.md`
maps each to the table it backs, records the split provenance of the CNN-MLPCoord row, and
documents a known discrepancy in Table 7's season rows.

## Known issues

- `evaluate.py`'s `summarize()` prints `%.4f`; should be `%.6g` (see caveat above).
- `third_party/ppcon/ppcon/config.py` and `scripts/train_ppcon_no_coord.py` hardcode
  `Path.home()` as the output root, inherited from PPCon. Retraining will write to the home
  directory rather than `results/` until that is changed.
- `CNNMLPCoord.forward` takes `(day_rad, year, lat, lon)` while `FourierBGC.forward` takes
  `(day_rad, lat, lon, year)`. Both match the call sites their checkpoints were trained
  under, so unifying them is a behaviour change, not a rename.

## History

The reorganization was done with `git mv`, so `git log --follow` traces most files back
through their renames. Two exceptions, where the content changed too much for git's rename
detection: `models/backbone.py` and `models/ppcon_no_coord.py`, which came from `models.py`
and `ppcon_no_scalar_eval.py` respectively. Their history is still in the repository, reachable
at the old paths:

    git log --all -- models.py
    git log --all -- ppcon_no_scalar_eval.py

The three commits that make up the reorganization are `a21c3d9` (snapshot), `afd3e87` (SLURM
scripts removed) and `a2eacbe` (the move itself).

## Provenance

Training was run on the University of North Carolina at Charlotte URC cluster under SLURM.
Those job scripts were removed during reorganization; the commands above are what they ran.
See commit `a21c3d9` for the originals.
