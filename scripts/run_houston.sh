#!/usr/bin/env bash
# SCDE on Houston 13 -> 18  (alpha = 2.0, class-conditional subspace)
# Usage: ./scripts/run_houston.sh            # all 5 seeds
#        SEEDS=333 ./scripts/run_houston.sh  # single seed
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/exp_common.sh"
cd "$(dirname "$HERE")" || exit 1
mkdir -p "$OUT/logs"

$PY -u train_sisa_det.py $COMMON $(hou_args) $METHOD \
    --seeds "$SEEDS" --output_dir "$OUT" --run_tag houston
