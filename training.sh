#!/bin/bash

python ./deeprl/agents/ddpg/learn.py --mode train --env 1 \
--hidden1 40 --hidden2 20 --warmup 10000 \
--sigma 0.25 --theta 0.05 --mu 0 \
--replay_max_size 10000 --batch_size 128 \
--target_update_rate 0.2 \
--validate_steps 0