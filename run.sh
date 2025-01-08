#!/bin/bash
temperature=$1
max_tokens=$2 
time_out=$3 
num_workers=$4 


# 默认设置
[ ! -n "$temperature"  ] && temperature=0.2
[ ! -n "$max_tokens" ] && max_tokens=20
[ ! -n "$time_out" ] && time_out=10
[ ! -n "$num_workers" ] && num_workers=50


echo temperature:$temperature
echo max_tokens:$max_tokens
echo time_out:$time_out
echo num_workers:$num_workers

python eval.py \
    --temperature "$temperature" \
    --max_tokens "$max_tokens" \
    --time_out "$time_out" \
    --num_workers "$num_workers"

if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
