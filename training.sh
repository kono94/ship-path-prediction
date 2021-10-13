#!/bin/bash

python ./deeprl/agents/ddpg/learn.py --mode train --resume runs --env 1 \
--hidden1 64 --hidden2 32 --warmup 1024 \
--sigma 0.7 --theta 0.2 --mu 0 \
--replay_max_size 50000 --batch_size 256 \
--tau 0.0001 --train_iter 50000