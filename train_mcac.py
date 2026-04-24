import torch
import torch.nn.functional as F
import gymnasium as gym
from model import Policy
from config import Config

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# actor outputs action logits; critic outputs a scalar state value V(s)
actor  = Policy(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim).to(device)
critic = Policy(obs_dim=cfg.obs_dim, action_dim=1).to(device)

actor_opt  = torch.optim.Adam(actor.parameters(),  lr=1e-3)
critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
env = gym.make(cfg.env_name)

for episode in range(cfg.n_episodes):
    obs, _ = env.reset()
    log_probs, values, rewards = [], [], []

    total_reward = 0.0

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # actor samples an action from the policy distribution
        dist  = torch.distributions.Categorical(logits=actor(obs_t))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        # critic estimates how good the current state is
        values.append(critic(obs_t).squeeze())

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        total_reward += reward
        if terminated or truncated:
            break

    # compute discounted returns G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    values    = torch.stack(values)
    log_probs = torch.stack(log_probs)

    # advantage = how much better the actual return was vs the critic's expectation
    # detach() stops gradients flowing into the critic when updating the actor
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # actor loss: reinforce the actions that had positive advantage, suppress negative ones
    actor_loss  = -(log_probs * advantages).sum()

    # critic loss: train the critic to better predict future returns
    critic_loss = F.mse_loss(values, returns)

    actor_opt.zero_grad();  actor_loss.backward();  actor_opt.step()
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

    if (episode + 1) % 50 == 0:
        print(f"episode {episode+1:4d}  avg_steps {len(rewards):3d} avg_rewards {total_reward}  "
              f"actor_loss {actor_loss.item():.2f}  critic_loss {critic_loss.item():.2f}")

torch.save(actor.state_dict(), "policy.pth")
print("saved policy.pth")
