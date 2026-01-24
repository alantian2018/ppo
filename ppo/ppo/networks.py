import torch
from torch.nn import Module, Linear, ReLU, Sequential, Conv2d, Flatten
from torch.distributions import Categorical


class Actor(Module):
    """Actor network -> policy pi(a|s)"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        if obs.dim==1:
            obs = obs.unsqueeze(0)

        return Categorical(logits=self.net(obs))


class Critic(Module):
    """Critic network -> value function V(s)"""
    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim==1:
            obs = obs.unsqueeze(0)

        return self.net(obs)


class CNNActor(Module):
    """CNN Actor network -> policy pi(a|s) for image observations."""
    def __init__(self, in_channels: int, height: int, width: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        # After conv with same padding, spatial dims preserved
        self.fc = Sequential(
            Flatten(),
            Linear(hidden_size * height * width, hidden_size),
            ReLU(),
            Linear(hidden_size, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        # obs: (batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # add batch dim
        obs = obs.permute(0, 3, 1, 2)  # HWC -> CHW
        x = self.conv(obs)
        logits = self.fc(x)
        return Categorical(logits=logits)


class CNNCritic(Module):
    """CNN Critic network -> value function V(s) for image observations."""
    def __init__(self, in_channels: int, height: int, width: int, hidden_size: int):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        self.fc = Sequential(
            Flatten(),
            Linear(hidden_size * height * width, hidden_size),
            ReLU(),
            Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2)
        x = self.conv(obs)
        return self.fc(x)