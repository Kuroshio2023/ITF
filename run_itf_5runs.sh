#!/bin/bash
set -e
cd ~/H-Sets
IMG_CS="${IMG_CS:-64}"
CUB_CS="${CUB_CS:-32}"
SEEDS="42 43 44 45 46"

run_model () {
    local gpu=$1 ds=$2 m=$3 cs=$4
    for seed in $SEEDS; do
        tag="seed${seed}_${ds}_${m}"
        echo "[GPU${gpu}] $(date +%H:%M) START $tag cs=$cs"
        CUDA_VISIBLE_DEVICES=$gpu python run_itf.py \
            --dataset "$ds" --model "$m" --num_samples 1000 \
            --hierarchy two-level --coarse_size "$cs" \
            --top_B 5 --block_size 8 --seed "$seed" --tag "$tag" \
            > "logs/itf_5runs_${tag}.log" 2>&1
        echo "[GPU${gpu}] $(date +%H:%M) END $tag (exit $?)"
    done
}

( run_model 0 imagenet vgg16              "$IMG_CS"; run_model 0 cub vgg16              "$CUB_CS" ) > logs/itf_5runs_gpu0.log 2>&1 &
( run_model 1 imagenet resnet101          "$IMG_CS"; run_model 1 cub resnet101          "$CUB_CS" ) > logs/itf_5runs_gpu1.log 2>&1 &
( run_model 2 imagenet densenet121        "$IMG_CS"; run_model 2 cub densenet121        "$CUB_CS" ) > logs/itf_5runs_gpu2.log 2>&1 &
( run_model 3 imagenet mobilenet_v3_large "$IMG_CS"; run_model 3 cub mobilenet_v3_large "$CUB_CS" ) > logs/itf_5runs_gpu3.log 2>&1 &

wait
echo "=== ALL 40 CONFIGS COMPLETE $(date) ==="
