#!/bin/bash

data_set=${1:-"full"}  # full数据集or sanitized数据集， 默认为 'full'
n=${2:-5}              # total number of samples in pass@k， 默认为 5
k=${3:-1}              # k in pass@k， 默认为 1
num_workers=${4:-2}    # 并发数，默认为 1
few_shot_id=$5         # 选取id为哪些的数据作为few shot，例如 2,3,4， 可选参数


echo "data_set:$data_set"
echo "n:$n"
echo "k:$k"
echo "num_workers:$num_workers"
echo "few_shot_id:$few_shot_id"


if [ -n "${few_shot_id[*]}" ]; then
    IFS=',' read -r -a few_shot_id <<< "${few_shot_id[*]}"
    python eval.py \
        --data_set "$data_set" \
        --n "$n" \
        --k "$k" \
        --num_workers "$num_workers" \
        --few_shot_id "${few_shot_id[@]}" 
else
    python eval.py \
    --data_set "$data_set" \
    --n "$n" \
    --k "$k" \
    --num_workers "$num_workers" 
fi


if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
