#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

num=$[($(echo $3 | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$3 $PYTHON -m torch.distributed.launch \
            --nproc_per_node=$num $(dirname "$0")/../tools/test.py \
            $CONFIG $CHECKPOINT --launcher pytorch --eval bbox ${@:4}
