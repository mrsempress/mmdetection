#!/usr/bin/env bash
# $1: config, $2: gpu ids, $3: dir_name (optional)

DIR_NAME=task_name
WORKERS_PER_GPU=-1

PYTHON=${PYTHON:-"python3"}
num=$[($(echo $2 | wc -L)+1)/2]
CUDA_VISIBLE_DEVICES=$2 $PYTHON -m torch.distributed.launch \
            --master_port=$((RANDOM + 10000)) \
            --nproc_per_node=$num $(dirname "$0")/../tools/train_ssl.py $1 \
            --dir=${3:-$DIR_NAME} \
            --workers_per_gpu=${4:-$WORKERS_PER_GPU} \
            --launcher pytorch
