"""Base agent interface for all RL algorithms."""

from abc import ABC, abstractmethod
import torch
import os


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, obs_size: int, n_actions: int, device: torch.device, config: dict):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = device
        self.config = config

    @abstractmethod
    def select_action(self, state, training: bool = True) -> int:
        """Choose an action given current state."""

    @abstractmethod
    def update(self, *args, **kwargs) -> dict:
        """Perform one learning update. Returns dict of metrics."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable state for saving."""

    @abstractmethod
    def load_state_dict(self, state: dict):
        """Restore agent from saved state."""

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.load_state_dict(data)
