#!/bin/bash

n_episodes=300
max_seed=5
remain_rate=0.3
env_id="BreakoutNoFrameskip-v4"

poetry run python src/tools/is_prune/train.py \
    --n_episodes $n_episodes \
    --max_seed $max_seed \
    --remain_rate $remain_rate \
    --env_id $env_id

n_episodes=600
max_seed=5
remain_rate=0.3
env_id="BreakoutNoFrameskip-v4"

poetry run python src/tools/is_prune/train.py \
    --n_episodes $n_episodes \
    --max_seed $max_seed \
    --remain_rate $remain_rate \
    --env_id $env_id

n_episodes=900
max_seed=5
remain_rate=0.3
env_id="BreakoutNoFrameskip-v4"

poetry run python src/tools/is_prune/train.py \
    --n_episodes $n_episodes \
    --max_seed $max_seed \
    --remain_rate $remain_rate \
    --env_id $env_id