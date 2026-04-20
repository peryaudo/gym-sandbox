import torch
import gymnasium as gym
from model import Policy
from config import Config

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy = Policy(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make(cfg.env_name)

for episode in range(cfg.n_episodes):
    obs, _ = env.reset()
    log_probs, rewards = [], []

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        dist = torch.distributions.Categorical(logits=policy(obs_t))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        if terminated or truncated:
            break

    G, returns = 0, []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -torch.stack(log_probs).dot(returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 50 == 0:
        print(f"episode {episode+1:4d}  steps {len(rewards):3d} loss {loss.item()}")

torch.save(policy.state_dict(), "policy.pth")
print("saved policy.pth")
