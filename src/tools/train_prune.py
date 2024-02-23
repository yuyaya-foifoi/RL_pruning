import gym
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.models.DQN.network import CNNQNetwork
from src.models.DQN.util import PrioritizedReplayBuffer
from src.pruning.slth.edgepopup import modify_module_for_slth

device = "cuda"
save_path = "./logs/test_0222/is_slth_save_dict.pkl"

env = gym.make("PongNoFrameskip-v4", full_action_space=True)
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
net = CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(
    device
)
target_net = CNNQNetwork(
    env.observation_space.shape, n_action=env.action_space.n
).to(device)
target_update_interval = (
    2000  # 学習安定化のために用いるターゲットネットワークの同期間隔
)
net = modify_module_for_slth(net, 0.3).to(device)
target_net = modify_module_for_slth(target_net, 0.3).to(device)

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
n_episodes = 300  # 学習を行うエピソード数


def update(batch_size, beta):
    obs, action, reward, next_obs, done, indices, weights = (
        replay_buffer.sample(batch_size, beta)
    )
    obs, action, reward, next_obs, done, weights = (
        obs.float().to(device),
        action.to(device),
        reward.to(device),
        next_obs.float().to(device),
        done.to(device),
        weights.to(device),
    )

    # 　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN.
        # ① 現在のQ関数でgreedyに行動を選択し,
        greedy_action_next = torch.argmax(net(next_obs), dim=1)
        # ②　対応する価値はターゲットネットワークのものを参照します.
        q_values_next = (
            target_net(next_obs)
            .gather(1, greedy_action_next.unsqueeze(1))
            .squeeze(1)
        )

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    # 　TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    replay_buffer.update_priorities(
        indices, (target_q_values - q_values).abs().detach().cpu().numpy()
    )

    return loss.item()


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
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())

        step += 1
        actions.append(action)

    print(
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

torch.save(save_dict, save_path)
