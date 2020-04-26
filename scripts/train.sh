#!/usr/bin/env bash
# $1: config, $2: gpu ids
# can use CUDA_LAUNCH_BLOCKING=1 to debug

num=$[($(echo $2 | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$2 python3 -m tools.train $1 \
                --gpus $num \
                --validate \
                --test \
                --dir //output/test
