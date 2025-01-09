#!/bin/bash
data_set=$1  # full数据集or sanitized数据集
n=$2  # total number of samples in pass@k
k=$3  # k in pass@k
num_workers=$4  # 并发数
few_shot_id=$5  # 选取id为哪些的数据作为few shot，例如 2,3,4


# 默认设置
[ -z "$data_set" ] &&data_set="full"
[ ! -n "$n"  ] && n=5
[ ! -n "$k" ] && k=1
[ ! -n "$num_workers" ] && num_workers=1
[ -z "$few_shot_id" ] &&data_set="2,3,4"


echo data_set:$data_set
echo n:$n
echo k:$k
echo num_workers:$num_workers
echo few_shot_id:$few_shot_id


python eval.py \
    --data_set "$data_set" \
    --few_shot_id "[$few_shot_id]" \
    --n "$n" \
    --k "$k" \
    --num_workers "$num_workers"

if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
