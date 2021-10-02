#!/bin/bash

python ./deeprl/agents/ddpg/learn.py --mode train --env 1 \
--hidden1 80 --hidden2 40 --warmup 10000 \
--sigma 0.3 --theta 0.15 --mu 0 \
--replay_max_size 50000 --batch_size 128 \
--tau 0.5 \
--validate_steps 0