#!/bin/bash
source ./setup.sh

EXPERIMENT_ID=coll_big_ships_S3_10secs_NEW_APPROACH
ALGO=bc
ENV=ais-v0
TRAIN_STEPS=30
STRUCTURE="[64,64]"
EXPERT_PATH=data/expert_trajectories/$EXPERIMENT_ID-ais_expert_trajectories.pickle

## SAMPLE EXPERT TRAJECTORIES (USUALLY ONCE)
python ./deeprl/ais_imitation.py --mode sample --expert_samples_path $EXPERT_PATH

#6 7 8 9
SEEDS=(5)
for i in "${SEEDS[@]}"
do
        SEED=$i
        prefix=experiments/$ALGO/$EXPERIMENT_ID#$STRUCTURE-steps$TRAIN_STEPS-seed$SEED
        POLICY_SAVE=$prefix/policy.pth
        # create folder (-p to create parent directories as needed)
        mkdir -p $prefix
        EVAL_PATH=$prefix/steps$TRAIN_STEPS#$STRUCTURE#seed$SEED.csv
        ## TRAIN ON EXPERT SAMPLES

        python ./deeprl/ais_imitation.py --mode train --algo $ALGO --env $ENV \
                --training_steps $TRAIN_STEPS --network $STRUCTURE \
            --policy_path $POLICY_SAVE --expert_samples_path $EXPERT_PATH --seed $SEED

        ## TEST THE TRAINED POLICY
        python ./deeprl/ais_imitation.py --mode test --env  $ENV --algo $ALGO --policy_path  $POLICY_SAVE  \
                       --animation_delay 0.15 --evaluation_path $EVAL_PATH  --render
done


