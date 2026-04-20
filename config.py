from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    obs_dim: int = 4
    action_dim: int = 2
    n_episodes: int = 500
