#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/ddpg/learn.py --mode train --resume runs --env 3 \
--hidden1 128 --hidden2 64 --warmup 128 \
--sigma 0.2 --theta 0.05 --mu 0 \
--replay_max_size 100000 --batch_size 128 \
--tau 0.001 --train_iter 70000 \
--actor_lr_rate 0.0001 --critic_lr_rate 0.001 --epsilon_max_decay 200000 \
--validate_episodes 0 --render