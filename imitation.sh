#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/curve_imitation.py --mode test --policy_path bc_policy.pth

#python ./deeprl/agents/curve_imitation.py --mode train --algo bc --env curve-v0 \
        --training_steps 50000 --hidden1 256 --hidden2 256 \
        --policy_path bc_policy.pth

