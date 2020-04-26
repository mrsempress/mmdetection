#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

num=$[($(echo $3 | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$3 python -m tools.test \
            $CONFIG $CHECKPOINT --eval bbox --show
