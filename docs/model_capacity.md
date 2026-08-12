# Model capacity reference

Trainable-parameter counts for every config in `configs/models/`, produced with:

```bash
python tools/param_count.py configs/models/*.yaml --num-classes 4
python tools/param_count.py configs/models/*.yaml --num-classes 3
```

Both runs build every config and forward a dummy batch (`(2, 1, 4096)` for
1D, `(2, 1, 64, 64)` for 2D) — all 26 configs build and forward
successfully at both class counts (exit code 0).

## 1D configs

| config | params (nc=3) | params (nc=4) |
|---|---:|---:|
| cnn1d_1x1 | 191 | 200 |
| cnn1d_2x2 | 215 | 232 |
| cnn1d_4x4 | 2279 | 3360 |
| cnn1d_multihead | 4295 | 4320 |
| cnn1d_res | 645 | 654 |
| cnn1d_se | 296 | 305 |
| cnn1d_eca | 200 | 209 |
| cnn1d_res_se | 750 | 759 |
| cnn1d_cbam | 338 | 347 |
| cnn1d_wide_plain | 2387 | 2420 |
| cnn1d_wide_res | 2421 | 2438 |
| cnn1d_wide_se | 2398 | 2423 |
| cnn1d_wide_eca | 2396 | 2429 |

## 2D configs

| config | params (nc=3) | params (nc=4) |
|---|---:|---:|
| cnn2d_1x1 | 359 | 368 |
| cnn2d_2x2 | 359 | 368 |
| cnn2d_4x4 | 2447 | 3528 |
| cnn2d_multihead | 4463 | 4488 |
| cnn2d_res | 1149 | 1158 |
| cnn2d_se | 464 | 473 |
| cnn2d_eca | 368 | 377 |
| cnn2d_res_se | 1254 | 1263 |
| cnn2d_cbam | 758 | 767 |
| cnn2d_wide_plain | 5171 | 5204 |
| cnn2d_wide_res | 5238 | 5256 |
| cnn2d_wide_se | 5165 | 5198 |
| cnn2d_wide_eca | 5180 | 5213 |

## Capacity-matched pairs

The `wide` family (channel progression starting at 8→16→32, roughly 4× the
width of the other families above) is the only family in `configs/models/`
built to be capacity-matched: `wide_plain` is the fixed reference per
dimensionality, and `wide_res`/`wide_se`/`wide_eca`'s channel counts were
tuned (module hyperparameters `r`, `k`, `blocks` left at their defaults) so
each lands within **±5%** of the reference's trainable-parameter count, at
both `num_classes=3` and `4`.

| dim | variant | channels (c1, c2, c3) | Δ params vs. `wide_plain` (nc=3) | Δ params vs. `wide_plain` (nc=4) |
|---|---|---|---:|---:|
| 1D | `wide_res` | 4, 9, 16 | +1.42% | +0.74% |
| 1D | `wide_se`  | 7, 13, 24 | +0.46% | +0.12% |
| 1D | `wide_eca` | 8, 16, 32 (unchanged) | +0.38% | +0.37% |
| 2D | `wide_res` | 5, 10, 17 | +1.30% | +1.00% |
| 2D | `wide_se`  | 7, 12, 32 | −0.12% | −0.12% |
| 2D | `wide_eca` | 8, 16, 32 (unchanged) | +0.17% | +0.17% |

All six pairs are within the ±5% tolerance at both class counts.
`wide_eca`'s channel counts are identical to `wide_plain`'s — ECA's
attention conv has only `k` weights per insertion (no bias, no linear
layers), which is negligible against a multi-thousand-parameter conv
encoder, so no tuning was needed. `res` and `se` needed meaningfully
smaller channel counts than the naive 8→16→32 progression: `res` adds a
second conv+BN per stage (its main path is two convs, not one) and `se`
adds two `Linear` layers per stage, both of which scale faster with
channel width than a single plain conv does.

**Not capacity-matched:** the pre-existing `_1x1`/`_2x2`/`_4x4`/`multihead`
families (which vary only aggregator/head settings, not encoder width) and
the original `res`/`se`/`eca`/`res_se`/`cbam` families (channel progression
1→2→4→8) predate this constraint and were never tuned against each other —
their parameter counts in the tables above differ by design, not by
oversight.
