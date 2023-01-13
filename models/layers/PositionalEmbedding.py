import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.unsqueeze(torch.arange(max_len), 0)
        self.register_buffer('position', position)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos_emb = self.embedding(self.position)
        x = x + pos_emb
        return self.dropout(x)
