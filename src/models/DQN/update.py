import torch


def update(
    net,
    target_net,
    optimizer,
    loss_func,
    replay_buffer,
    device,
    batch_size,
    beta,
    gamma,
):
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

    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        greedy_action_next = torch.argmax(net(next_obs), dim=1)
        q_values_next = (
            target_net(next_obs)
            .gather(1, greedy_action_next.unsqueeze(1))
            .squeeze(1)
        )
    target_q_values = reward + gamma * q_values_next * (1 - done)

    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    replay_buffer.update_priorities(
        indices, (target_q_values - q_values).abs().detach().cpu().numpy()
    )

    return loss.item()
