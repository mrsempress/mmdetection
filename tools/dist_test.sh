#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

num=$[($(echo $GPUS | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
    --master_port 23456 --nproc_per_node=$num $(dirname "$0")/test.py \
    $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
