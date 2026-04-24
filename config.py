from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    obs_dim: int = 4
    action_dim: int = 2
    n_episodes: int = 500

# @dataclass
# class Config:
#     env_name: str = "LunarLander-v3"
#     obs_dim: int = 8
#     action_dim: int = 4
#     n_episodes: int = 5000
