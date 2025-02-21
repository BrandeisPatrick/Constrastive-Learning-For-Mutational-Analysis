import torch
import torch.nn as nn
import math

def initialize_weights(module):
    """Initialize weights using Kaiming uniform."""
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)


class PositionalEncoding(nn.Module):
    def __init__(self, embd_dim, max_seq_len=1200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, embd_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embd_dim, 2).float() * (-math.log(10000.0) / embd_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embd_dim)
        self.register_buffer('pe', pe)  # Non-trainable buffer

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class TokenEmbedding(nn.Module):
    def __init__(self, n_token, embd_dim):
        super(TokenEmbedding, self).__init__()
        self.embd_dim = embd_dim
        self.token_embedder = nn.Embedding(n_token, embd_dim)
        
    def forward(self, x):
        return self.token_embedder(x) * math.sqrt(self.embd_dim)
    