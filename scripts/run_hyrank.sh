#!/usr/bin/env bash
# SCDE on HyRANK Dioni -> Loukia  (alpha = 2.0, sisa_blocks = 24)
# Usage: ./scripts/run_hyrank.sh
#        SEEDS=333 ./scripts/run_hyrank.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/exp_common.sh"
cd "$(dirname "$HERE")" || exit 1
mkdir -p "$OUT/logs"

$PY -u train_sisa_det.py $COMMON $(yhr_args) $METHOD \
    --seeds "$SEEDS" --output_dir "$OUT" --run_tag hyrank
