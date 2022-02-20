#!/bin/bash
source ./setup.sh

python ./deeprl/scripts/curve_imitation.py --mode test --env curve-heading-speed-v0 --policy_path  bc_policy_heading_speed_circle_512_512.pth --animation_delay 0.15

#python ./deeprl/scripts/curve_imitation.py --mode train --algo bc --env curve-heading-speed-v0 \
        --training_steps 100000 --hidden1 512 --hidden2 512 \
        --policy_path bc_policy_heading_speed_circle_512_512.pth --seed 5

