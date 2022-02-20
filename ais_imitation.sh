#!/bin/bash
source ./setup.sh

N_SAMPLES=28941
ALGO=bc
ENV=ais-v0
N_NEURONS=32
TRAIN_STEPS=10
EXPERT_PATH=data/expert_trajectories/ais_expert_trajectories_$N_SAMPLES.pickle


## SAMPLE EXPERT TRAJECTORIES (USUALLY ONCE)
#python ./deeprl/scripts/ais_imitation.py --mode sample --expert_samples_path $EXPERT_PATH --n_samples $N_SAMPLES

#6 7 8 9
SEEDS=(5 6 7 8 9)
for i in "${SEEDS[@]}"
do
        SEED=$i
        prefix=experiments/$ALGO/neurons$N_NEURONS-steps$TRAIN_STEPS-samples$N_SAMPLES-seed$SEED
        POLICY_SAVE=$prefix/policy.pth
        # create folder (-p to create parent directories as needed)
        mkdir -p $prefix
        EVAL_PATH=$prefix/$ALGO#steps$TRAIN_STEPS#neurons$N_NEURONS#seed$SEED.csv
        ## TRAIN ON EXPERT SAMPLES

        # python ./deeprl/scripts/ais_imitation.py --mode train --algo $ALGO --env $ENV \
        #         --training_steps $TRAIN_STEPS --hidden1 $N_NEURONS --hidden2 $N_NEURONS \
        #         --policy_path $POLICY_SAVE --expert_samples_path $EXPERT_PATH --seed $SEED

        ## TEST THE TRAINED POLICY
        python ./deeprl/scripts/ais_imitation.py --mode test --env  $ENV --algo $ALGO --policy_path  $POLICY_SAVE  \
                        --animation_delay 0.15 --evaluation_path $EVAL_PATH


done


