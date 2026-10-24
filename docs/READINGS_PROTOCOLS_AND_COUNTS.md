# Identifiability readings — protocols and per-class counts

This note accompanies the identifiability-readings table of the paper. Per the
reviewer's request, the long table note was removed from the typeset table and
is documented here instead.

Table numbers below are as of version 9 of the manuscript (`SCDE-v-09`): the
readings table is Table 1, the comparison table is Table 5, and the scanned
$\alpha$ table is Table 3. If tables are renumbered in a later version, the
reference is the readings table, the comparison table, and the $\alpha$-scan
table respectively.

## 1. The three directional protocols

The row "Directional consistency" aggregates three independent measurements,
listed in increasing order of assumption strength:

| Tag | Protocol | What is measured |
|:--|:--|:--|
| (a) | First three principal angles to the analytic form of the extended linear mixing model (ELMM) | Agreement between the estimated subspace and the direction implied by the physical model |
| (b) | First three principal directions across cities acquired with one sensor | Agreement between two independent source-side estimates |
| (c) | First principal angle in the synthetic setting only | Agreement when the true offset is known by construction |

(a) and (b) are read on the real benchmarks; (c) is reported for the synthetic
setting only and is **not** extrapolated to the real data.

## 2. Containment and the per-class counts

"Containment in $\hat{S}$" reports the energy fraction of the cross-domain
offset that falls inside the estimated subspace. The value $45\%$–$97\%$ is
aggregated across classes; the class-wise comparison against the
same-dimensional random baseline is:

| Benchmark | Classes where containment exceeds the same-dimension random baseline |
|:--|:--|
| Pavia | 7 / 7 |
| HyRANK | 11 / 11 |
| Houston | 5 / 7 |

Houston is the most uneven of the three, and its two non-exceeding classes are
the reason the aggregate range is reported as a range rather than a single
value.

## 3. Stability

"Estimation stability" is the largest principal angle between the subspaces
estimated under different block partitions, swept from $4\times4$ to
$32\times32$ blocks. The same-dimension random baseline is computed by drawing
random subspaces of the same dimension and repeating the sweep.

## 4. The scanned magnitude $\alpha^*$

$\alpha^*$ is the measured ratio of the cross-domain offset magnitude to the
source-internal offset magnitude. It is a measurement, not a tuned value, and
it **need not coincide** with the optimum of the $\alpha$ scan reported in
Table 3 of the paper, which is taken under a fixed blocking and is
benchmark-dependent. On HyRANK the scan optimum sits at a larger $\alpha$ than
the production configuration; the production value is the one used for every
reported result.

## 5. Coverage bound

$\varepsilon$ is defined in Eq. (4) of the paper. The three values
($0.743$ / $0.625$ / $0.996$ for Pavia / HyRANK / Houston) are to be read
against their own same-dimension random baselines ($0.975$ / $0.986$ /
$0.947$). On Houston the bound is **not** below the random baseline, so no
bound-based argument is made for that benchmark.
