#!/bin/bash
echo "starting"
export PYTHONPATH=$(pwd)
SEED=5
ALGO=bc
ENV=ais-v0
N_SAMPLES=30
N_NEURONS=512
TRAIN_STEPS=2048
prefix=experiments/$ALGO/neurons$N_NEURONS-steps$TRAIN_STEPS-seed$SEED
# create folder (-p to create parent directories as needed)
mkdir -p $prefix
POLICY_SAVE=$prefix/policy.pth
EXPERT_PATH=experiments/ais_expert_trajectories_$N_SAMPLES.pickle
EVAL_PATH=$prefix/$ALGO#steps$TRAIN_STEPS#neurons$N_NEURONS#seed$SEED.csv
## SAMPLE EXPERT TRAJECTORIES (USUALLY ONCE)
#python ./deeprl/agents/ais_imitation.py --mode sample --expert_samples_path $EXPERT_PATH --n_samples $N_SAMPLES


## TRAIN ON EXPERT SAMPLES
#python ./deeprl/agents/ais_imitation.py --mode train --algo $ALGO --env $ENV \
        --training_steps $TRAIN_STEPS --hidden1 $N_NEURONS --hidden2 $N_NEURONS \
        --policy_path $POLICY_SAVE --expert_samples_path $EXPERT_PATH --seed $SEED


## TEST THE TRAINED POLICY
python ./deeprl/agents/ais_imitation.py --mode test --env  $ENV --policy_path  $POLICY_SAVE  --n_samples $N_SAMPLES \
                --animation_delay 0.15 --evaluation_path $EVAL_PATH
