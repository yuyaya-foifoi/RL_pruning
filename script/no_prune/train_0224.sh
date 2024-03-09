#!/bin/bash

n_episodes=600
max_seed=5
env_id="PongNoFrameskip-v4"

poetry run python src/tools/no_prune/train.py \
    --n_episodes $n_episodes \
    --max_seed $max_seed \
    --env_id $env_id

n_episodes=900
max_seed=5
env_id="PongNoFrameskip-v4"

poetry run python src/tools/no_prune/train.py \
    --n_episodes $n_episodes \
    --max_seed $max_seed \
    --env_id $env_id