# FourierBGC

Code for *Fourier Feature Encoding for Spatiotemporal Inputs in CNN-Based
Reconstruction of Biogeochemical Argo Profiles*.

This repository contains FourierBGC, in both its broadcast and learned-projection
variants, alongside the other models used in that paper's ablation study, which was
designed to isolate the effect of the coordinate encoding from that of the training
recipe. It is meant to be read alongside the paper.

```bash
pip install -r requirements.txt
```

## Layout

```
data/            PPCon's published train/test splits; every model reads from here except PPCon-NoCoord's training, which reads the identical copy vendored under third_party/
models/          all model variants, one file each, plus the shared backbone
helpers/         dataset loader, Fourier basis, z-score constants, seeding
scripts/         the two training entry points that are not train.py
results/         all 105 checkpoints behind the paper's tables
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
python train.py --model cnn_no_coord --target_var NITRATE --epochs 100 --seed 0
```

Works for `cnn_no_coord`, `cnn_raw_coord`, `cnn_mlp_coord` and `fourierbgc_broadcast`.
The other two have their own entry points, because FourierBGC computes its projection
inside the model and PPCon-NoCoord uses PPCon's recipe rather than the common one:

```bash
python scripts/train_fourierbgc.py --target_var NITRATE --epochs 100 --seed 0
python scripts/train_ppcon_no_coord.py --variable NITRATE --epochs 100 --seed 0
```

PPCon-NoCoord uses PPCon's per-variable epoch counts: 100 / 150 / 125 for
NITRATE / CHLA / BBP700.

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

FourierBGC also has a standalone evaluator, equivalent to the above:

```bash
python scripts/evaluate_fourierbgc.py --target_var CHLA \
    --checkpoint results/fourierbgc/seed0/CHLA/best.pt
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
