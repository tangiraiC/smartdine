import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_idx, item_idx):
        u = self.user_factors(user_idx)
        i = self.item_factors(item_idx)
        dot = (u * i).sum(dim=-1)
        bu = self.user_bias(user_idx).squeeze(-1)
        bi = self.item_bias(item_idx).squeeze(-1)
        return self.global_bias + dot + bu + bi
