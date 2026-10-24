#!/usr/bin/env bash
# Z0 baseline: the same backbone and training protocol with both SCDE signals
# switched off. Subspace estimation and view generation are not run at all.
#
# Usage: ./scripts/run_z0.sh houston|hyrank|pavia
#        SEEDS=333 ./scripts/run_z0.sh houston
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/exp_common.sh"
cd "$(dirname "$HERE")" || exit 1
BENCH=${1:-houston}
mkdir -p "$OUT/logs"

case "$BENCH" in
  houston) ARGS=$(hou_args) ;;
  hyrank)  ARGS=$(yhr_args) ;;
  pavia)   ARGS=$(pav_args) ;;
  *) echo "usage: $0 houston|hyrank|pavia"; exit 2 ;;
esac

$PY -u train_sisa_det.py $COMMON $ARGS \
    --sisa_lambda 0 --sdc_lambda 0 \
    --seeds "$SEEDS" --output_dir "$OUT" --run_tag z0_$BENCH
