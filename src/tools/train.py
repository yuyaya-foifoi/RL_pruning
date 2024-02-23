import os

import click
import gym
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.models.DQN.network import CNNQNetwork
from src.models.DQN.update import update
from src.models.DQN.util import PrioritizedReplayBuffer
from src.pruning.slth.edgepopup import modify_module_for_slth
from src.utils.date import get_current_datetime_for_path
from src.utils.logger import setup_logger
from src.utils.seed import torch_fix_seed


@click.command()
@click.option("--eval_steps", default=1_000, help="Number of eval steps.")
@click.option("--n_episodes", default=300, help="Number of steps.")
@click.option("--seed", default=0, help="seed")
@click.option(
    "--is_prune",
    type=click.BOOL,
    default=True,
    help="whether net will be pruned",
)
@click.option("--remain_rate", default=0.3, help="whether net will be pruned")
@click.option("--env_id", default="PongNoFrameskip-v4", help="Environment ID.")
def main(eval_steps, n_episodes, seed, is_prune, remain_rate, env_id):

    device = "cuda"
    current_date = get_current_datetime_for_path()

    if is_prune:
        flg = "is_prune_seed_"
    else:
        flg = "is_not_prune_seed_"

    save_dir = "./logs/{}/{}/{}".format(current_date, env_id, flg + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(save_dir)
    logger.info(save_dir)

    torch_fix_seed(seed)

    env = gym.make(env_id, full_action_space=True)
    # env = gym.make("PooyanNoFrameskip-v4", full_action_space=True)
    # Atari preprocessing wrapper
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    # Frame stacking
    env = gym.wrappers.FrameStack(env, 4)

    """
        リプレイバッファの宣言
    """
    buffer_size = 100000  # 　リプレイバッファに入る経験の最大数
    initial_buffer_size = 10000  # 学習を開始する最低限の経験の数
    replay_buffer = PrioritizedReplayBuffer(buffer_size)

    """
        ネットワークの宣言
    """
    net = CNNQNetwork(
        env.observation_space.shape, n_action=env.action_space.n
    ).to(device)
    target_net = CNNQNetwork(
        env.observation_space.shape, n_action=env.action_space.n
    ).to(device)
    target_update_interval = (
        2000  # 学習安定化のために用いるターゲットネットワークの同期間隔
    )
    if is_prune:
        logger.info("the model will be pruned")
        net = modify_module_for_slth(net, remain_rate).to(device)
        target_net = modify_module_for_slth(target_net, remain_rate).to(device)

    """
        オプティマイザとロス関数の宣言
    """
    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # オプティマイザはAdam
    loss_func = nn.SmoothL1Loss(
        reduction="none"
    )  # ロスはSmoothL1loss（別名Huber loss）

    """
        Prioritized Experience Replayのためのパラメータβ
    """
    beta_begin = 0.4
    beta_end = 1.0
    beta_decay = 500000
    # beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
    beta_func = lambda step: min(
        beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay)
    )

    """
        探索のためのパラメータε
    """
    epsilon_begin = 1.0
    epsilon_end = 0.01
    epsilon_decay = 50000
    # epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
    epsilon_func = lambda step: max(
        epsilon_end,
        epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay),
    )

    """
        その他のハイパーパラメータ
    """
    gamma = 0.99  # 　割引率
    batch_size = 32
    n_episodes = n_episodes  # 学習を行うエピソード数

    step = 0
    rewards = []
    actions = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # ε-greedyで行動を選択
            obs_np = np.array(obs)
            # NumPy配列をPyTorchのテンソルに変換
            obs = torch.tensor(obs_np, dtype=torch.float)
            action = net.act(obs.to(device), epsilon_func(step))
            # 環境中で実際に行動
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            next_obs_np = np.array(next_obs)
            # NumPy配列をPyTorchのテンソルに変換
            next_obs = torch.tensor(next_obs_np, dtype=torch.float)

            # リプレイバッファに経験を蓄積
            replay_buffer.push([obs, action, reward, next_obs, done])
            obs = next_obs
            obs = obs.cpu()

            # ネットワークを更新
            if len(replay_buffer) > initial_buffer_size:
                update(
                    net,
                    target_net,
                    optimizer,
                    loss_func,
                    replay_buffer,
                    device,
                    batch_size,
                    beta_func(step),
                    gamma,
                )

            # ターゲットネットワークを定期的に同期させる
            if (step + 1) % target_update_interval == 0:
                target_net.load_state_dict(net.state_dict())

            step += 1
            actions.append(action)

        logger.info(
            "Episode: {},  Step: {},  Reward: {}".format(
                episode + 1, step + 1, total_reward
            )
        )
        rewards.append(total_reward)

    save_dict = {
        "actions": actions,
        "rewards": rewards,
        "model_state": net.state_dict(),
    }

    torch.save(save_dict, os.path.join(save_dir, "save_dict.pkl"))


if __name__ == "__main__":
    main()
