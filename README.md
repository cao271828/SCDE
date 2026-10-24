# SCDE — Subspace-Constrained Domain Extension

Reference implementation of

> **Subspace-Constrained Domain Extension for Single-Source Cross-Scene
> Hyperspectral Classification**

SCDE is a training-time augmentation scheme for single-source cross-scene
hyperspectral classification. It estimates, from the heterogeneity *inside* the
source scene only, the linear subspace in which the cross-scene spectral offset
falls, generates displaced views along that subspace, and constrains the
representation to be invariant along it. Nothing is added at inference time.

Two components:

| Component | Role | Code |
|:--|:--|:--|
| **SISA** | Estimates the offset subspace from source-internal spatial blocks, then draws displaced views along it | `estimate_shift_subspace()`, `sisa_augment()` |
| **SDC** | Supervised contrastive constraint on the resulting view pair | `supcon_shift_loss()` |

---

## 1. Requirements

Python 3.10+, one CUDA GPU.

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.x on a single RTX 4090 (24 GB). The default batch size is
4096 and the largest benchmark needs roughly 20 GB, so a 24 GB card is
comfortable; 16 GB may need `--batch_size` reduced, which changes the reported
numbers.

## 2. Data

Nothing is downloaded automatically. Put the files in `./data` (or point
`SCDE_DATA` at your own directory):

```
data/
├── Houston13.mat  Houston13_7gt.mat
├── Houston18.mat  Houston18_7gt.mat
├── paviaU.mat     paviaU_7gt.mat
├── paviaC.mat     paviaC_7gt.mat
└── HyRANK/
    ├── Dioni.tif   Dioni_GT.tif
    └── Loukia.tif  Loukia_GT.tif
```

* Houston: the two scenes are reduced to their overlapping channels and aligned
  to 209 x 955 pixels, with 7 common classes.
* Pavia: the two scenes are reduced to their overlapping channels (102 bands)
  and spatially aligned, with 7 classes.
* HyRANK: the two label systems are mapped onto 12 shared groups, of which 11
  occur in the target scene.

The `.mat` keys are read by `datasets.py` (`ori_data` / `paviaU` / `map`), the
GeoTIFFs by `custom_datasets.py`. Adjust those two files if your copies use
different keys.

## 3. Reproduce

```bash
chmod +x scripts/*.sh

./scripts/run_houston.sh          # 5 seeds
./scripts/run_hyrank.sh
./scripts/run_pavia.sh
SEEDS=333 ./scripts/run_houston.sh    # one seed, ~1 min for a smoke test
./scripts/run_z0.sh houston           # no-augmentation baseline
```

Results land in `./results/<source>2<target>_<tag>_s<seed>_results.json`; the
terminal prints `OA mean +- std` over the seeds.

Expected best OA (%) over seeds `{333,444,555,666,777}` — manuscript Table III:

| Benchmark | SCDE | Z0 |
|:--|--:|--:|
| Houston 13 -> 18 | 83.85 | 76.17 |
| HyRANK Dioni -> Loukia | 70.82 | 62.01 |
| Pavia UP -> PC | 84.05 | 80.88 |

### 3.1 Reference configuration

The three-arm configuration table was moved out of the manuscript to save a
page; this is its authoritative copy. Common settings: patch `13x13`, batch
`4096`, `200` epochs, `lambda_sisa = 0.5`, `lambda_sdc = 1.0`, `tau = 0.1`,
subspace dimension `r = 5`, in-block class threshold `m = 10`, seeds
`{333, 444, 555, 666, 777}`.

| Benchmark | alpha | lr | wd | re_ratio | Dedicated setting | Params |
|:--|--:|--:|--:|--:|:--|--:|
| Houston 13 -> 18 | 2.0 | 0.01 | 1e-4 | 5 | `--sisa_classwise` | 0.7255 M |
| HyRANK Dioni -> Loukia | 2.0 | 0.001 | 1e-3 | 1 | `--sisa_blocks 24` | 0.7977 M |
| Pavia UP -> PC | 0.02 | 0.01 | 1e-4 | 5 | exclusive GPU | 0.7509 M |

The exact flags are in `scripts/exp_common.sh`; the three arm functions
`hou_args`, `yhr_args` and `pav_args` are the rows of the table above.

## 4. Manuscript to code

| Manuscript | Code |
|:--|:--|
| III-A, per-class intra-block deviation `d_{b,c}`, block deviation `d_b` | `estimate_shift_subspace()`, block loop |
| III-A, row-centred matrix `D~`, SVD, `V_S`, `sigma_k` | same function, `np.linalg.svd` |
| III-A, class-conditional variant | `estimate_shift_subspace(..., classwise=True)` |
| III-B, `delta = alpha * sum_k eps_k sigma_k v_k`, spatially constant | `sisa_augment()`, `mask_mode='none'` |
| III-C, `z = W_e f(X)`, `L2`-normalised embeddings | `VCA_Res2DCNN.forward()` (`fc_embed`), `F.normalize` |
| III-C, SupCon form, `tau = 0.1` | `supcon_shift_loss()` |
| III-D, `L = CE + lambda_sisa * L_SISA + lambda_sdc * L_SDC` | `train_one_seed()`, loss assembly |
| III-D, inference path `argmax W_c f(X)` | `train_one_seed()`, evaluation loop |
| Backbone | `vca_dres.VCA_Res2DCNN` |
| IV, protocol, alpha, lr, wd, re_ratio, seeds | `scripts/exp_common.sh` |
| III-E, protocols behind the identifiability readings | `docs/READINGS_PROTOCOLS_AND_COUNTS.md` |

## 5. Deviations from the manuscript text

These are known differences between the v-09 manuscript text and this code. The
code is what produced the reported numbers, so the text, not the code, should be
corrected.

1. **SDC subsampling threshold.** The text says the pairs are subsampled when
   `2N > 1024`. The code caps the number of anchors at `--sdc_max_n 1024`, i.e.
   it subsamples when `N > 1024` (2048 embeddings).
2. **`L_SDC` normalisation.** The text writes `1/(2N) * sum_i`. The code averages
   over anchors that have at least one positive, which is identical whenever
   every anchor does.
3. **Block grid.** The text says `n_b = 8 x 8`. The code derives the block size
   as `H // n_blocks` and `W // n_blocks`, so the grid is
   `ceil(H/bh) x ceil(W/bw)`; at `n_blocks = 8` this is 9 x 9 on Houston and
   Pavia. `n_b` in the `sigma_k` normalisation is the number of blocks actually
   used after the `|Omega_{b,c}| >= m` filter.
4. **`sigma_k`.** The text normalises by `sqrt(n_b)`; the code divides the
   singular values by `sqrt(number of blocks used)` and applies no Bessel
   correction.
5. **HyRANK class count.** The text says 11 common classes. The label map keeps
   12 groups; 11 of them occur in the target scene, which is why the target-side
   support is 11.
6. **Backbone.** The manuscript only says the backbone is based on the LDGnet
   skeleton with a modified and optimised design. Concretely: a frozen identity
   spectral projection, three residual blocks (32/64/128 channels), and two
   heads on the shared 512-d feature (a 128-d embedding head for SDC and a
   K-way linear classifier for SISA and the main loss).
7. **Flags not listed in the paper's configuration sentence.** `--proj_init Identity`, `--no_lr_decay`,
   `--gpu_resident 1` and the data-repetition factor `--re_ratio` are part of
   the protocol. `--nsd --nsd_mode swa` only adds an extra averaged-weight
   read-out after training; it does not change the reported best OA.

## 6. Verification

```bash
SCDE_DATA=/path/to/data ./scripts/verify_release.sh            # all three arms
SCDE_DATA=/path/to/data ./scripts/verify_release.sh houston    # one arm
```

The script runs three checks: that every module imports and the backbone
builds, that the parameter counts match the manuscript (0.7255 / 0.7977 /
0.7509 M), and that seed 333 of each arm reproduces the published OA
(83.58 / 71.18 / 82.77) within `TOL` percentage points, default 0.5. The
tolerance covers the last decimal, which depends on the GPU and the PyTorch
build.

## 7. Notes

* The offset subspace is computed once from source data before training, with
  `numpy` linear algebra and no gradients.
* `lambda_sisa` and `lambda_sdc` default to `0.5` and `1.0` as in the
  manuscript; setting both to `0` reproduces `Z0`.
* No target-domain sample enters the optimisation. Target labels are used only
  to select the best epoch, following the protocol of the compared methods.
* **Scope of the released script.** `train_sisa_det.py` is the research
  training script. Besides the SCDE path it still carries options from earlier
  experiments (alternative projection initialisations, directional controls,
  spatial masks, extra regularisers). All of them are inactive under the flags
  in `scripts/exp_common.sh` and are scheduled for removal in the cleaned-up
  release.
* This repository accompanies a paper under review. Citation details will be
  added on publication.

## License

MIT, see [LICENSE](LICENSE).

