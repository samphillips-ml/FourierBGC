# FourierBGC

Code for *Fourier Feature Encoding for Spatiotemporal Inputs in CNN-Based Reconstruction of
Biogeochemical Argo Profiles* (Phillips & Li).

A controlled ablation over how spatiotemporal coordinates are represented, holding PPCon's
convolutional backbone fixed. Seven models differ only in the coordinate encoding and, for two of
them, the training procedure.

> **Note.** This file documents the repository as it stands before the reorganization. Paths change
> when `models/`, `helpers/` and `results/` land; this file is updated in the same commit.

## Model name map

The manuscript and the code use different names. This is the mapping.

| Paper | `--model` flag | Class | Entry point |
|---|---|---|---|
| PPCon | `ppcon` | (vendored) | `evaluate.py` only, released checkpoint |
| PPCon-NoCoord | `ppcon_no_scalar` | — | `ppcon_no_scalar/run.py` |
| CNN-NoCoord | `cnn` | `RawCNNProbe` | `train.py` |
| CNN-RawCoord | `cnn_scalar` | `RawCNNScalarProbe` | `train.py` |
| CNN-MLPCoord | `cnn_mlpcoord` | `CNNMLPCoordProbe` | `train.py` |
| FourierBGC-Broadcast | `fourierbgc_with_year` | `FourierBGCBroadcast` | `train.py` |
| FourierBGC | `fourierbgc_learned` | `FourierBGC` | `fourier_learned_projection/run_train_with_year.py` |

`fourierbgc` (21 channels, no year) and the two `transformer*` flags are earlier variants that do not
appear in the manuscript.

## Data

`data/{NITRATE,CHLA,BBP700}/float_ds_sf_{train,test}.csv`, inherited unmodified from PPCon's
published splits. Not reprocessed here. Every model reads from this one location.

## Training

All models we train ourselves use the common recipe: Adam, lr 1e-3, 5-epoch linear warmup then
cosine decay, gradient clipping at max norm 1.0, 100 epochs, batch size 32, five seeds (0-4).

    # CNN-NoCoord, CNN-RawCoord, CNN-MLPCoord, FourierBGC-Broadcast
    python train.py --model {cnn|cnn_scalar|cnn_mlpcoord|fourierbgc_with_year} \
        --target_var {NITRATE|CHLA|BBP700} --epochs 100 --seed {0..4} \
        --save_dir results/{model}/seed{N}/{VAR}/model

    # FourierBGC (separate entry point)
    python fourier_learned_projection/run_train_with_year.py \
        --target_var {VAR} --epochs 100 --seed {N} --save_dir {dir}

PPCon-NoCoord uses PPCon's own recipe (Adadelta, lr 1.0, three-term loss, no clipping or schedule)
and PPCon's per-variable epoch counts, 100 / 150 / 125 for NITRATE / CHLA / BBP700:

    python ppcon_no_scalar/run.py --variable {VAR} --epochs {100|150|125} \
        --seed {N} --save_dir {dir}

PPCon itself is never retrained; the authors' released checkpoint is evaluated directly.

## Evaluation

Reproduces PPCon's own RMSE methodology (per-profile RMSE, then averaged within region and season
buckets). Region and season definitions follow PPCon's released evaluation code.

    python evaluate.py --model {flag} --target_var {VAR} --checkpoint {dir}/best.pt

    # the two checkpoint-directory models
    python evaluate.py --model ppcon --target_var {VAR} \
        --checkpoint_dir ppcon_baseline/ppcon/results/{VAR}/{date}/model --epoch {100|150|125}
    python evaluate.py --model ppcon_no_scalar --target_var {VAR} \
        --checkpoint_dir {dir}/model --epoch {100|150|125}

    # FourierBGC
    python fourier_learned_projection/run_evaluate_with_year.py \
        --target_var {VAR} --checkpoint {dir}/best.pt

**Caveat:** `evaluate.py` prints RMSE with `%.4f`. For bbp700, whose values are around 2e-4, that is
a single significant figure and every printed value collapses to `0.0002`. Read bbp700 at full
precision, not from this output.

## Reproducing the manuscript's numbers

The 105 checkpoints behind every table are deposited to Zenodo, not tracked here; GMD's code and data
policy requires a DOI-issuing archive and states GitHub is unsuitable for archiving frozen versions.
`RELEASE.md` maps each table cell to its archived checkpoint.

Note that CNN-MLPCoord's published row draws on two different training batches: chlorophyll-*a* and
nitrate from one, bbp700 from the other. A separate model is trained per target, so this is
well-defined, but neither batch alone reproduces the row.

## Provenance of this repository

Training was run on the University of North Carolina at Charlotte URC cluster under SLURM. Those job
scripts are removed as of this commit; the commands above are what they ran. See commit `a21c3d9`
for the originals.
