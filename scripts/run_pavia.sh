#!/usr/bin/env bash
# SCDE on Pavia UP -> PC  (alpha = 0.02)
# Usage: ./scripts/run_pavia.sh
#        SEEDS=333 ./scripts/run_pavia.sh
#
# Note: on this benchmark the displacement is very small, so keep AMP off
# (the released script does not enable it) and run one job per GPU.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/exp_common.sh"
cd "$(dirname "$HERE")" || exit 1
mkdir -p "$OUT/logs"

$PY -u train_sisa_det.py $COMMON $(pav_args) $METHOD \
    --seeds "$SEEDS" --output_dir "$OUT" --run_tag pavia
