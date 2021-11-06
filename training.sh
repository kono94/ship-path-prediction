#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)

# python ./deeprl/agents/ddpg/learn.py --mode test --file_prefix runs/4_400_300_35_42


python ./deeprl/agents/ddpg/learn.py --mode train --env 4 \
--hidden1 400 --hidden2 300 --warmup 256 \
--sigma 0.4 --theta 0.1 --mu 0 --gamma 0 \
--replay_max_size 100000 --batch_size 256 \
--tau 0.001 --train_iter 200000  --reward_barrier 38 --step_barrier 40000 \
--actor_lr_rate 0.001 --critic_lr_rate 0.001 --epsilon_max_decay 200000 \
--validate_episodes 3 &

python ./deeprl/agents/ddpg/learn.py --mode train --env 4 \
--hidden1 400 --hidden2 302 --warmup 256 \
--sigma 0.6 --theta 0.1 --mu 0 --gamma 0 \
--replay_max_size 100000 --batch_size 256 \
--tau 0.001 --train_iter 200000  --reward_barrier 38 --step_barrier 40000 \
--actor_lr_rate 0.0001 --critic_lr_rate 0.0001 --epsilon_max_decay 300000 \
--validate_episodes 3 &


python ./deeprl/agents/ddpg/learn.py --mode train --env 1 \
--hidden1 400 --hidden2 300 --warmup 256 \
--sigma 0.5 --theta 0.1 --mu 0 --gamma 0 \
--replay_max_size 100000 --batch_size 256 \
--tau 0.001 --train_iter 200000  --reward_barrier 19 --step_barrier 40000 \
--actor_lr_rate 0.0005 --critic_lr_rate 0.0005 --epsilon_max_decay 300000 \
--validate_episodes 3 &

