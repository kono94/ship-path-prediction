#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/ddpg/learn.py --mode test --resume runs --env 1 \
--hidden1 40 --hidden2 20 --warmup 1500 \
--sigma 0.5 --theta 0.1 --mu 0 \
--replay_max_size 100000 --batch_size 128 \
--tau 0.001 --train_iter 100000 \
--actor_lr_rate 0.0001 --critic_lr_rate 0.0001 --epsilon_max_decay 80000 --render