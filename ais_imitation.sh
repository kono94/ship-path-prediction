#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/ais_imitation.py --mode test --env  ais-v0 --policy_path  bc_policy_ais_512_512.pth --animation_delay 0.15

#python ./deeprl/agents/ais_imitation.py --mode train --algo bc --env ais-v0 \
        --training_steps 10000 --hidden1 512 --hidden2 512 \
        --policy_path bc_policy_ais_512_512.pth --seed 5
