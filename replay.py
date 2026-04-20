import torch
import gymnasium as gym
from model import Policy
from config import Config

cfg = Config()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
policy = Policy(obs_dim=cfg.obs_dim, action_dim=cfg.action_dim).to(device)
policy.load_state_dict(torch.load("policy.pth", map_location=device))
policy.eval()

env = gym.make(cfg.env_name, render_mode="human")
obs, _ = env.reset()
while True:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    action = policy(obs_t).argmax().item()
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
