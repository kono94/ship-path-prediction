#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

#python ./deeprl/agents/curve_imitation.py --mode test --env curve-heading-v0 --policy_path bc_policy_heading_128_128.pth --animation_delay 0.15

python ./deeprl/agents/curve_imitation.py --mode train --algo bc --env curve-heading-v0 \
        --training_steps 50000 --hidden1 256 --hidden2 256 \
      --policy_path bc_policy_heading_256_256.pth

