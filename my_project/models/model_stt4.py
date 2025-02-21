import torch
import torch.nn as nn
from .model_utils import *

class Model_STT4(nn.Module):
    def __init__(self, args):
        super(Model_STT4, self).__init__()
        self.args = args
        self.embd_dim = args.embd_dim
        self.hidn_dim = args.hidn_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len
        self.dropout = args.dropout
        self.n_token = args.n_token
        
        # Embedding and Positional Encoding
        self.enc_te = TokenEmbedding(n_token=self.n_token, embd_dim=self.embd_dim)
        self.enc_pe = PositionalEncoding(embd_dim=self.embd_dim, max_seq_len=self.max_seq_len, dropout=self.dropout)
        self.enc_e = nn.Sequential(self.enc_te, self.enc_pe)
        
        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embd_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidn_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder0 = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
                
        # MLP Module
        self.module0 = nn.Sequential(
            nn.Linear(self.embd_dim, self.embd_dim),
            nn.GELU(),
            nn.LayerNorm(self.embd_dim),
            nn.Linear(self.embd_dim, self.embd_dim),
            nn.LayerNorm(self.embd_dim)
        )
        
        # Distance Metrics
        self.edist1 = nn.PairwiseDistance(p=1, eps=1e-12)
        self.edist2 = nn.PairwiseDistance(p=2, eps=1e-12)
        self.csdist = nn.CosineSimilarity(dim=-1)
        
        # Convolution and Classification Head
        ndus = 3
        nreps = 3
        nd = ndus * nreps
        self.convol10 = nn.Sequential(
            nn.BatchNorm1d(nd),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=self.dropout)
        )
        t = self.max_seq_len * nd
        self.compress_dense = nn.Sequential(
            nn.Linear(t, t),
            nn.GELU(),
            nn.LayerNorm(t),
            nn.Dropout(p=self.dropout),
            nn.Linear(t, t // 2),
            nn.Dropout(p=self.dropout),
            nn.LayerNorm(t // 2),
            nn.Linear(t // 2, 2),
            nn.Sigmoid()
        )
        
        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        initialize_weights(self)

    def get_pad_mask(self, matrix):
        """
        Generates padding mask.
        Args:
            matrix: Tensor of shape (batch_size, seq_len)
        Returns:
            mask: Tensor of shape (batch_size, seq_len)
        """
        return (matrix == self.args.pad_token)

    def dist(self, e1, e2):
        """
        Computes distance metrics between two embeddings.
        Args:
            e1: Tensor of shape (batch_size, seq_len, embd_dim)
            e2: Tensor of shape (batch_size, seq_len, embd_dim)
        Returns:
            d: Tensor of shape (batch_size, seq_len, 3)
        """
        d1 = self.edist1(e1, e2)
        d2 = self.edist2(e1, e2)
        d3 = self.csdist(e1, e2)
        d = torch.stack([d1, d2, d3], dim=-1)
        return d

    def forward(self, src1, src2):
        """
        Forward pass of the model.
        Args:
            src1: Tensor of shape (batch_size, seq_len)
            src2: Tensor of shape (batch_size, seq_len)
        Returns:
            output_cls: Tensor of shape (batch_size, 2)
        """
        # Create padding masks
        src_pad_mask1 = self.get_pad_mask(src1)
        src_pad_mask2 = self.get_pad_mask(src2)
        
        # Embedding
        embedded1 = self.enc_e(src1)  # Shape: (batch_size, seq_len, embd_dim)
        embedded2 = self.enc_e(src2)  # Shape: (batch_size, seq_len, embd_dim)
        
        # Initial Distance
        dVec0 = [self.dist(embedded1, embedded2)]  # List of tensors
        
        # Encode and process
        memory1 = self.encoder0(embedded1, src_key_padding_mask=src_pad_mask1)
        memory1 = self.module0(memory1)
        memory2 = self.encoder0(embedded2, src_key_padding_mask=src_pad_mask2)
        memory2 = self.module0(memory2)
        dVec0.append(self.dist(memory1, memory2))
        
        # Additional Encoding Layer (if needed)
        memory1 = self.encoder0(memory1, src_key_padding_mask=src_pad_mask1)
        memory1 = self.module0(memory1)
        memory2 = self.encoder0(memory2, src_key_padding_mask=src_pad_mask2)
        memory2 = self.module0(memory2)
        dVec0.append(self.dist(memory1, memory2))
        
        # Concatenate Distances
        dVec0 = torch.cat(dVec0, dim=2)  # Shape: (batch_size, seq_len, 9)
        
        # Permute for Convolution
        dVec0 = dVec0.permute(0, 2, 1)  # Shape: (batch_size, 9, seq_len)
        
        # Convolution and Classification
        distCompress0 = self.convol10(dVec0)  # Shape: (batch_size, 9 * seq_len)
        output_cls = self.compress_dense(distCompress0)  # Shape: (batch_size, 2)
        
        return output_cls