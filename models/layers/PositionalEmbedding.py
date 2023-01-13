import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.unsqueeze(torch.arange(max_len), 0)
        self.register_buffer('position', position)
        self.embedding = nn.Embedding(max_len, d_model)

    def get_embedding(self,):
        return self.embedding(self.position)

    def forward(self, x):
        pos_emb = self.get_embedding()
        x = x + pos_emb
        return self.dropout(x)
