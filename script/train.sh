#!/bin/bash

n_episodes=300
seed=0
env_id="PongNoFrameskip-v4"

poetry run python src/tools/train.py \
    --n_episodes $n_episodes \
    --seed $seed \
    --is_prune False\
    --env_id $env_id

n_episodes=300
seed=0
remain_rate=0.3
env_id="PongNoFrameskip-v4"

poetry run python src/tools/train.py \
    --n_episodes $n_episodes \
    --seed $seed \
    --is_prune True\
    --remain_rate $remain_rate \
    --env_id $env_id