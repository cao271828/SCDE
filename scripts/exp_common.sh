#!/usr/bin/env bash
# SCDE - shared configuration for the three benchmarks.
#
# Every value below is the one used for the numbers reported in the manuscript.
# These settings were Table II of an earlier draft; they now live in README.md
# Sect. 3.1. Do not change them if you want to reproduce the published numbers.
#
# Usage: see run_houston.sh / run_hyrank.sh / run_pavia.sh

export KMP_DUPLICATE_LIB_OK=TRUE
PY=${PY:-python}

# Data root. Override with: SCDE_DATA=/path/to/data ./scripts/run_houston.sh
DATA_ROOT=${SCDE_DATA:-./data}

SEEDS=${SEEDS:-333,444,555,666,777}
OUT=${OUT:-./results}

# Settings shared by all three benchmarks.
COMMON="--patch_size 13 --epochs 200 --batch_size 4096 --no_lr_decay \
--gpu_resident 1 --num_workers 0 --proj_init Identity \
--nsd --nsd_mode swa --nsd_start 50"

# SCDE training signals. Use "--sisa_lambda 0 --sdc_lambda 0" for the Z0 baseline.
METHOD="--sisa_lambda 0.5 --sdc_lambda 1.0 --sdc_temp 0.1 --sdc_max_n 1024 \
--sisa_rank 5 --sisa_min_px 10"

hou_args() {
  echo "--source Houston13 --target Houston18 --data_path ${DATA_ROOT}/ \
--re_ratio 5 --lr 0.01 --wd 0.0001 --eval_batch_size 4096 \
--sisa_alpha 2.0 --sisa_blocks 8 --sisa_classwise"
}

yhr_args() {
  echo "--source Dioni --target Loukia --data_path ${DATA_ROOT}/HyRANK/ \
--re_ratio 1 --lr 0.001 --wd 0.001 --eval_batch_size 4096 \
--sisa_alpha 2.0 --sisa_blocks 24"
}

pav_args() {
  echo "--source paviaU --target paviaC --data_path ${DATA_ROOT}/ \
--re_ratio 5 --lr 0.01 --wd 0.0001 --eval_batch_size 4096 \
--sisa_alpha 0.02 --sisa_blocks 8"
}
