#!/usr/bin/env bash
# Release verification for the SCDE code package.
#
# Three checks, in order:
#   1. the modules import and the backbone builds for each benchmark
#   2. the parameter counts match the manuscript (0.7255 / 0.7977 / 0.7509 M)
#   3. one seed (333) of each arm reproduces the published OA
#
# Check 3 compares against the seed-333 readings of the manuscript runs with a
# tolerance of TOL percentage points (default 0.5), since the last decimal
# depends on the GPU and the PyTorch build.
#
# Usage:
#   SCDE_DATA=/root/data ./scripts/verify_release.sh            # all three arms
#   SCDE_DATA=/root/data ./scripts/verify_release.sh houston    # one arm
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/exp_common.sh"
cd "$(dirname "$HERE")" || exit 1

PY=${PY:-python}
TOL=${TOL:-0.5}
OUT=${OUT:-./results_verify}
mkdir -p "$OUT"

ARM=${1:-all}
case "$ARM" in
  all)     ARMS="houston hyrank pavia" ;;
  houston|hyrank|pavia) ARMS="$ARM" ;;
  *) echo "usage: $0 [all|houston|hyrank|pavia]"; exit 2 ;;
esac

expected_oa() {
  case "$1" in
    houston) echo 83.58 ;;   # SCDE, seed 333
    hyrank)  echo 71.18 ;;   # SCDE, seed 333
    pavia)   echo 82.77 ;;   # SCDE, seed 333
  esac
}
expected_params() {
  case "$1" in
    houston) echo 0.7255 ;;
    hyrank)  echo 0.7977 ;;
    pavia)   echo 0.7509 ;;
  esac
}

echo "================ 1/3 imports ================"
$PY - <<'EOF'
import sys
import datasets, custom_datasets, utils_HSI, aug_abundance, vca_dres, train_sisa_det
print("imports OK: train_sisa_det, vca_dres, datasets, custom_datasets, utils_HSI, aug_abundance")
EOF
[ $? -ne 0 ] && { echo "FAIL: imports"; exit 1; }

echo "================ 2/3 parameter counts ================"
$PY - "$DATA_ROOT" <<'EOF'
import sys
from datasets import get_dataset
from vca_dres import build_einit_cnn
root = sys.argv[1] if len(sys.argv) > 1 else './data'
spec = [('Houston13', 7, 'houston'), ('Dioni', 12, 'hyrank'), ('paviaU', 7, 'pavia')]
for name, cls, tag in spec:
    folder = root + ('/HyRANK/' if tag == 'hyrank' else '/')
    r = get_dataset(name, target_folder=folder)
    img, gt = r[0], r[1]
    m, k = build_einit_cnn(img, gt, img.shape[2], cls, seed=333, proj_init='Identity')
    p = sum(t.numel() for t in m.parameters()) / 1e6
    print(f"{tag:8s} bands={img.shape[2]:4d} k={k:4d} params={p:.4f} M")
EOF
[ $? -ne 0 ] && { echo "FAIL: parameter counts"; exit 1; }

echo "================ 3/3 one-seed reproduction ================"
rc=0
for a in $ARMS; do
  case "$a" in
    houston) ARGS=$(hou_args) ;;
    hyrank)  ARGS=$(yhr_args) ;;
    pavia)   ARGS=$(pav_args) ;;
  esac
  echo "---- $a (seed 333) ----"
  $PY -u train_sisa_det.py $COMMON $ARGS $METHOD \
      --seeds 333 --output_dir "$OUT" --run_tag "verify_$a" \
      > "$OUT/${a}_s333.log" 2>&1
  OA=$($PY - "$OUT" "$a" <<'EOF'
import glob, json, os, sys
out, arm = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(out, "*_s333_results.json")), key=os.path.getmtime)
if not fs:
    print("NA"); raise SystemExit
d = json.load(open(fs[-1], encoding='utf-8'))
v = d.get('best_OA', d.get('OA'))
print("NA" if v is None else f"{float(v) * 100:.2f}")
EOF
)
  EXP=$(expected_oa "$a")
  if [ "$OA" = "NA" ]; then
    echo "  $a: no result file, see $OUT/${a}_s333.log"; rc=1
  else
    D=$(awk -v a="$OA" -v b="$EXP" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.2f", d}')
    if awk -v d="$D" -v t="$TOL" 'BEGIN{exit !(d<=t)}'; then
      echo "  $a: OA=$OA  expected=$EXP  |diff|=$D  PASS"
    else
      echo "  $a: OA=$OA  expected=$EXP  |diff|=$D  FAIL (tolerance $TOL)"; rc=1
    fi
  fi
done
echo
[ $rc -eq 0 ] && echo "VERIFY: PASS" || echo "VERIFY: FAIL"
exit $rc
