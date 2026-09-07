# FourierBGC

Code for *Fourier Feature Encoding for Spatiotemporal Inputs in CNN-Based
Reconstruction of Biogeochemical Argo Profiles*.

This repository contains FourierBGC, in both its broadcast and learned-projection
variants, alongside the other models used in that paper's ablation study, which was
designed to isolate the effect of the coordinate encoding from that of the training
recipe. A full explanation of differences is in the paper.

```bash
pip install -r requirements.txt
```

## Layout

```
data/            PPCon's published train/test splits; every model reads from here except PPCon-NoCoord's training, which reads the identical copy vendored under third_party/
models/          all model variants, one file each, plus the shared backbone
helpers/         dataset loader, Fourier basis, z-score constants, seeding
scripts/         PPCon-NoCoord's trainer, which uses PPCon's recipe not ours
results/         all 105 checkpoints behind the paper's tables. 5 seeds for every variant.
third_party/     the PPCon baseline, vendored unmodified (MIT)
visualizations/  figure code and the figures themselves
archive/         superseded runs; not tracked, not needed
```

## Models

The paper and the code use different names.

| Paper | `--model` | File |
|---|---|---|
| PPCon | `ppcon` | `third_party/ppcon/` (released checkpoint, never retrained) |
| PPCon-NoCoord | `ppcon_no_coord` | `models/ppcon_no_coord.py` |
| CNN-NoCoord | `cnn_no_coord` | `models/cnn_no_coord.py` |
| CNN-RawCoord | `cnn_raw_coord` | `models/cnn_raw_coord.py` |
| CNN-MLPCoord | `cnn_mlp_coord` | `models/cnn_mlp_coord.py` |
| FourierBGC-Broadcast | `fourierbgc_broadcast` | `models/fourierbgc_broadcast.py` |
| FourierBGC | `fourierbgc` | `models/fourierbgc.py` |

`--target_var` is `NITRATE`, `CHLA` or `BBP700` throughout. Seeds 0-4 were used.

## Training

```bash
python train.py --model fourierbgc --target_var NITRATE --epochs 100 --seed 0
```

Trains any of the five models under the common recipe. Checkpoints land in
`results/{model}/seed{N}/{VAR}/`.

PPCon-NoCoord is the exception: it uses PPCon's own recipe rather than the common one,
with PPCon's per-variable epoch counts of 100 / 150 / 125, so it has its own script.

```bash
python scripts/train_ppcon_no_coord.py --variable NITRATE --epochs 100 --seed 0
```

PPCon itself is never retrained; its released checkpoint is evaluated directly.

## Evaluation

Reproduces PPCon's own RMSE methodology, using their region and season definitions.
Five of the seven models take a checkpoint file:

```bash
python evaluate.py --model fourierbgc --target_var CHLA \
    --checkpoint results/fourierbgc/seed0/CHLA/best.pt
```

PPCon and PPCon-NoCoord save one file per network per epoch, so they take a directory
and an epoch instead:

```bash
python evaluate.py --model ppcon --target_var CHLA \
    --checkpoint_dir results/ppcon/CHLA/model --epoch 150

python evaluate.py --model ppcon_no_coord --target_var CHLA \
    --checkpoint_dir results/ppcon_no_coord/seed0/CHLA/model --epoch 150
```

To reproduce a published table entry, evaluate all five seeds at once and get mean
and standard deviation for the pooled figure and every region and season:

```bash
python evaluate.py --model fourierbgc --target_var CHLA --seeds 0,1,2,3,4
```

## Checkpoints

`results/` holds all 105 checkpoints behind the paper's tables, at
`results/{model}/seed{N}/{VAR}/`. `results/MANIFEST.md` maps each to the table it
backs and records which training batch each published value came from.

## Known issues

- `third_party/ppcon/ppcon/config.py` and `scripts/train_ppcon_no_coord.py` hardcode
  `Path.home()` as the output root, inherited from PPCon. Retraining writes to your
  home directory, not `results/`.
- `scripts/train_ppcon_no_coord.py` defaults to `--epochs 0`, also inherited from
  PPCon's CLI. Run it without an explicit `--epochs` and you get an untrained model.
- The `eval_best.txt` files already in `results/` were written before RMSE printing
  was fixed to full precision, so their bbp700 values all read `0.0002`. Re-running
  the evaluation gives the real numbers.

## License

MIT. The bundled PPCon baseline (`third_party/ppcon/`) is MIT, Pietropolli et al.
