#!/bin/bash
source ./setup.sh

NEURONS=(64)


HIDDEN1=64
HIDDEN2=32
SEED=7
PREFIX=bc_S3_A2_$HIDDEN1#$HIDDEN2

python ./deeprl/scripts/curve_imitation.py --mode train --algo bc --env curve-heading-speed-v0 \
--training_steps 10000 --hidden1 $HIDDEN1 --hidden2 $HIDDEN2 \
--policy_path $PREFIX.pth  --seed $SEED

python ./deeprl/scripts/curve_imitation.py --mode test --algo bc --env curve-heading-speed-v0 --policy_path  $PREFIX.pth --animation_delay 0.1 --evaluation_path $PREFIX