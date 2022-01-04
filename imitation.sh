#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/curve_imitation.py --mode test --env curve-heading-speed-v0 --policy_path  bc_policy_heading_speed_128_128.pth --animation_delay 0.15

#python ./deeprl/agents/curve_imitation.py --mode train --algo bc --env curve-heading-speed-v0 \
        --training_steps 30000 --hidden1 128 --hidden2 128 \
      --policy_path bc_policy_heading_speed_128_128.pth

