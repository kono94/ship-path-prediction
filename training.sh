#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

python ./deeprl/agents/ddpg/learn.py --mode test --resume runs --env 4 \
--hidden1 400 --hidden2 300 --warmup 256 \
--sigma 0.3 --theta 0.05 --mu 0 --gamma 0.99 \
--replay_max_size 100000 --batch_size 256 \
--tau 0.001 --train_iter 200000 \
--actor_lr_rate 0.0005 --critic_lr_rate 0.0005 --epsilon_max_decay 200000 \
--validate_episodes 3 --render