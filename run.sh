#!/bin/bash

num_workers=${1:-16}    # 并发数，默认为 1

echo num_workers:$num_workers

python eval.py --num_workers $num_workers

if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
