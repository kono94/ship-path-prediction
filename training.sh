#!/bin/bash

python ./deeprl/agents/ddpg/learn.py --mode test --resume runs --env 1 \
--hidden1 300 --hidden2 150 --warmup 10000 \
--sigma 0.25 --theta 0.05 --mu 0 \
--replay_max_size 50000 --batch_size 128 \
--tau 0.5 \
--validate_steps 1