# recommender/models/tower.py

import torch
import torch.nn as nn


class TowerMLP(nn.Module):
    """
    Simple MLP tower for a single modality (text, image, or dense features).

    Args:
        input_dim (int): dimensionality of the input feature vector
                         e.g., 768 for text/image embeddings, 8 for dense feats.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        returns: [batch_size] predicted ratings
        """
        return self.net(x).squeeze(-1)
