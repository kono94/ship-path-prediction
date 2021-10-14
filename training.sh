#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/ddpg/learn.py --mode train --resume runs --env 4 \
--hidden1 4 --hidden2 2 --warmup 64 \
--sigma 0.4 --theta 0.15 --mu 0 --gamma 0 \
--replay_max_size 100000 --batch_size 64 \
--tau 0.001 --train_iter 8000 \
--actor_lr_rate 0.0001 --critic_lr_rate 0.0001 --epsilon_max_decay 6000 \
--validate_episodes 0