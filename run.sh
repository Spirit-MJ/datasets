#!/bin/bash
eval_target=${1:-"strict"}  # 选择严格或宽松 strict or loose
num_workers=${2:-16}    # 并发数，默认为 1


echo eval_target:$eval_target
echo num_workers:$num_workers

python eval.py --eval_target $eval_target --num_workers $num_workers

if [ $? -eq 0 ]; then
    echo "Tests completed successfully."
else
    echo "There was an error during the tests."
fi
