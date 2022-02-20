#!/bin/bash
source ./setup.sh

python ./deeprl/agents/ddpg/learn.py --mode train --resume runs --env 3 \
--hidden1 400 --hidden2 300 --warmup 256 \
--sigma 0.3 --theta 0.05 --mu 0 \
--replay_max_size 100000 --batch_size 256 \
--tau 0.001 --train_iter 100000 \
--actor_lr_rate 0.00015 --critic_lr_rate 0.0005 --epsilon_max_decay 200000 \
--validate_episodes 0 