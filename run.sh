#!/bin/bash
system_prompt=$1
temperature=$2 
max_tokens=$3 
time_out=$4 
num_workers=$5 


# 默认设置
[ -z "$system_prompt" ] && system_prompt="you are a helpful assistant"
[ ! -n "$temperature"  ] && temperature=0.2
[ ! -n "$max_tokens" ] && max_tokens=20
[ ! -n "$time_out" ] && time_out=10
[ ! -n "$num_workers" ] && num_workers=50


echo system_prompt:$system_prompt
echo temperature:$temperature
echo max_tokens:$max_tokens
echo time_out:$time_out
echo num_workers:$num_workers

python eval.py \
    --system_prompt "$system_prompt" \
    --temperature "$temperature" \
    --max_tokens "$max_tokens" \
    --time_out "$time_out" \
    --num_workers "$num_workers"

if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
