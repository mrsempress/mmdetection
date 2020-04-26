#!/usr/bin/env bash
# $1: config, $2: checkpoint, $3: gpu ids

num=$[($(echo $3 | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$3 python -m tools.test $1 $2 --gpus $num --eval bbox ${@:4}
