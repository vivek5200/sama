"""Specialist decision heads. Each is single-task."""

import torch
import torch.nn as nn


class ChoiceHead(nn.Module):
    head_type = "choice"

    def __init__(self, hidden_dim: int, proj_dim: int = 512,
                 max_options: int = 8, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(proj_dim, max_options)

    def forward(self, h):
        return self.head(self.proj(h))


class NoulHead(nn.Module):
    head_type = "noul"

    def __init__(self, hidden_dim: int, proj_dim: int = 512, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(proj_dim, 1)

    def forward(self, h):
        return self.head(self.proj(h)).squeeze(-1)


class ScoreHead(nn.Module):
    head_type = "score"

    def __init__(self, hidden_dim: int, proj_dim: int = 512,
                 max_levels: int = 8, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(proj_dim, max_levels)

    def forward(self, h):
        return self.head(self.proj(h))


HEAD_CLASSES = {
    "choice": ChoiceHead,
    "noul": NoulHead,
    "score": ScoreHead,
}


def load_head(head_type: str, hidden_dim: int, state_dict_path: str, device):
    cls = HEAD_CLASSES[head_type]
    head = cls(hidden_dim=hidden_dim)
    state = torch.load(state_dict_path, map_location="cpu")
    head.load_state_dict(state)
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    return head
