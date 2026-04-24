import torch
import gymnasium as gym
from model import Policy
from config import Config

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy = Policy(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make(cfg.env_name)

EPISODE_BATCH_SIZE = 4

for episode in range(0, cfg.n_episodes, EPISODE_BATCH_SIZE):
    all_log_probs, all_returns = [], []

    total_reward = 0.0

    for _ in range(EPISODE_BATCH_SIZE):
        obs, _ = env.reset()
        log_probs, rewards, returns = [], [], []

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            dist = torch.distributions.Categorical(logits=policy(obs_t))
            action = dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            total_reward += reward

            if terminated or truncated:
                break

        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        all_log_probs.extend(log_probs)
        all_returns.extend(returns)

    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=device)
    all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

    loss = -torch.stack(all_log_probs).dot(all_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_steps = len(all_log_probs) / EPISODE_BATCH_SIZE
    total_reward /= EPISODE_BATCH_SIZE
    if episode % 50 == 0:
        print(f"episode {episode:4d} avg_steps {avg_steps} avg_reward {total_reward} loss {loss.item()}")

torch.save(policy.state_dict(), "policy.pth")
print("saved policy.pth")
