#!/bin/bash
source ./setup.sh

SEEDS=(5 6 7 8 9)

for i in "${SEEDS[@]}"
do
        SEED=$i
        PREFIX=ddpg_S4_A2_R3#$SEED

        python ./deeprl/scripts/curve_imitation.py --mode train --algo ddpg --env curve-heading-speed-distance-v0 \
        --training_steps 30000 --hidden1 400 --hidden2 300 \
        --policy_path $PREFIX.pth  --seed $SEED

        python ./deeprl/scripts/curve_imitation.py --mode test --algo ddpg --env curve-heading-speed-distance-v0 --policy_path  $PREFIX.pth --animation_delay 0.1 --evaluation_path $PREFIX
done
