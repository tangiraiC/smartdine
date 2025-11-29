import torch
import torch.nn as nn

class JointRecModel(nn.Module):
    def __init__(self, n_users, text_dim=768, img_dim=768, dense_dim=8,
                 user_emb_dim=64, hidden_dims=(512, 256)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_emb_dim)

        input_dim = user_emb_dim + text_dim + img_dim + dense_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, text, img, dense):
        u = self.user_emb(user_idx)              # (B, user_emb_dim)
        x = torch.cat([u, text, img, dense], dim=-1)
        out = self.mlp(x).squeeze(-1)            # (B,)
        return out
