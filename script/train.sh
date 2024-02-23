#!/bin/bash

n_episodes=300
seed=0
is_prune=False
remain_rate=None
env_id="PongNoFrameskip-v4"

poetry run python src/tools/train.py \
    --n_episodes $n_episodes \
    --seed $seed \
    --is_prune $is_prune  \
    --env_id $env_id

n_episodes=300
seed=0
is_prune=True
prune_rate=0.3
env_id="PongNoFrameskip-v4"

poetry run python src/tools/train.py \
    --n_episodes $n_episodes \
    --seed $seed \
    --is_prune $is_prune  \
    --remain_rate $remain_rate \
    --env_id $env_id