# Provenance manifest

Every checkpoint behind every table in the manuscript. 105 files: seven models,
three target variables, five seeds each (PPCon excepted — one released checkpoint
per variable, not retrained).

A separate model is trained per target variable, so each `{model}/seed{N}/{VAR}`
cell is independent. That is what makes the split provenance below well-defined
rather than an inconsistency.

## Layout

    results/{model}/seed{N}/{VAR}/best.pt            five common-recipe models
    results/ppcon_no_coord/seed{N}/{VAR}/model/      model_conv_{epoch}.pt
    results/ppcon/{VAR}/model/                       model_{conv,day,year,lat,lon}_{epoch}.pt

`eval_best.txt` sits beside each checkpoint where the original run produced one.
**Do not read bbp700 out of those files** — they were written with `%.4f`, which at
2e-4 is one significant figure, so every bbp700 value in them reads `0.0002`.

## Where each model's checkpoints came from

| Paper model | results/ | Original location | Selection |
|---|---|---|---|
| PPCon | `ppcon/` | `third_party/ppcon/ppcon/results/{VAR}/{date}/model/` | released checkpoint, epoch 100/150/125 |
| PPCon-NoCoord | `ppcon_no_coord/` | `archive/repo--local-output/ppcon_no_scalar/` | final epoch, 100/150/125 |
| CNN-NoCoord | `cnn_no_coord/` | `archive/repo--cluster-output/cnn/` | `best.pt` |
| CNN-RawCoord | `cnn_raw_coord/` | `archive/repo--cluster-output/cnn_scalar/` | `best.pt` |
| CNN-MLPCoord | `cnn_mlp_coord/` | **two batches, see below** | `best.pt` |
| FourierBGC-Broadcast | `fourierbgc_broadcast/` | `archive/repo--cluster-output/fourierbgc_with_year/` | `best.pt` |
| FourierBGC | `fourierbgc/` | `archive/repo--cluster-output/fourier_learned_projection/` | `best.pt` |

## CNN-MLPCoord: split provenance

CNN-MLPCoord was trained twice on 10 Aug 2026, seven hours apart. Nitrate is
byte-identical between the two batches; chlorophyll-*a* and bbp700 seeds 0, 1 and 2
differ. The published Table 4 row draws on **both**:

| Variable | Batch used | Archive path | Evidence |
|---|---|---|---|
| NITRATE | either (identical) | `archive/repo--more--cnn_mlp_coord-run2/` | both give 0.3293 (0.0066) |
| CHLA | run 2 | `archive/repo--more--cnn_mlp_coord-run2/` | run 2 gives 0.0673 (0.0005) = Table 4; run 1 gives 0.0676 (0.0002) |
| BBP700 | run 1 | `archive/repo--cnn_mlpcoord-run1/` | run 1 gives 1.7606 (0.0098) = Table 4; run 2 gives 1.7658 (0.0167) |

Both batches are preserved in `archive/` in full. Neither alone reproduces the
published row, which is why this manifest exists.

## Known discrepancy: Table 7's season rows

Re-evaluating bbp700 at full precision on 7 Sep 2026 reproduced Table 4 exactly for
all seven models, and reproduced Tables 5 and 6 (chlorophyll-*a*) exactly. It did
**not** reproduce the four season rows of Table 7.

Those rows fail an internal consistency check the region rows pass: seasons cover
932/948 profiles, so their n-weighted mean must approximate the pooled value.

| Model | Published seasons average to | Published pooled | Gap |
|---|---|---|---|
| CNN-MLPCoord | 1.635 | 1.761 | 0.126 |
| FourierBGC-Broadcast | 1.517 | 1.595 | 0.078 |
| FourierBGC | 1.546 | 1.641 | 0.095 |

Recomputed from these checkpoints, all three land within 0.002 of pooled. Every
qualitative conclusion in Sect. 6.5 is unaffected: the lowest model per subset is
unchanged in all nine subsets, the projection's sign is unchanged in all four
seasons, and "improves every region in summer, degrades every region in winter"
holds 5/5 in both at region×season resolution.

## Verification, 7 Sep 2026

Every checkpoint here was re-evaluated through the reorganized code and checked against the
manuscript. All 21 model × variable cells of Table 4 reproduce.

Eighteen match at full precision. Three differ by one unit in the last printed digit, and the
cause is an aggregation convention rather than a discrepancy: **the published Table 4 values
are the mean of the per-seed values after each was rounded to 4 dp by `eval_best.txt`**, not
the rounded mean of full-precision values. Applying the paper's own convention reproduces all
three exactly.

| Cell | Mean of full precision | Mean of rounded per-seed | Published |
|---|---|---|---|
| CNN-RawCoord / NITRATE | 0.3412 | 0.3411 | 0.3411 |
| CNN-MLPCoord / CHLA | 0.0672 | 0.0673 | 0.0673 |
| FourierBGC-Broadcast / NITRATE | 0.2338 | 0.2339 | 0.2339 |

The differences are ~1e-4, an order of magnitude below the seed standard deviations, and change
no reported comparison, percentage or conclusion.

One residual: for FourierBGC-Broadcast / NITRATE, seed 2's recorded value is 0.2372 where
re-evaluation gives 0.23707 (rounding to 0.2371). Every other recorded value matches
re-evaluation bit-for-bit. A single seed differing at 1e-4 is consistent with the original
having been evaluated on the cluster GPU and the check on CPU.
