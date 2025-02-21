import torch
import torch.nn as nn
from .model_utils import TokenEmbedding, PositionalEncoding, initialize_weights

class Model_STT4_encoder(nn.Module):
    def __init__(self, args):
        """
        Modified model for contrastive learning.
        This model uses an embedding layer with positional encoding, a Transformer encoder,
        and an MLP. It processes two input sequences (src1 and src2), pools over the sequence
        dimension, and returns two embedding vectors.

        Args:
            args: An object with the following attributes:
                - embd_dim: Embedding dimension.
                - hidn_dim: Hidden dimension for feedforward layers.
                - n_heads: Number of attention heads.
                - n_layers: Number of Transformer encoder layers.
                - dropout: Dropout probability.
                - n_token: Vocabulary size (number of tokens).
                - max_seq_len: Maximum sequence length.
                - pad_token: The token used for padding.
        """
        super(Model_STT4_encoder, self).__init__()
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
            batch_first=True
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
        
        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        initialize_weights(self)

    def get_pad_mask(self, matrix):
        """
        Generates a padding mask.
        
        Args:
            matrix: Tensor of shape (batch_size, seq_len)
        
        Returns:
            mask: Tensor of shape (batch_size, seq_len)
        """
        return (matrix == self.args.pad_token)

    def _process(self, src):
        """
        Process a single input sequence through embedding, Transformer encoder, MLP,
        and mean pooling while ignoring pad tokens.
        
        Args:
            src: Tensor of shape (batch, seq_len)
        
        Returns:
            embedding: Tensor of shape (batch, embd_dim)
        """
        # Create padding mask if available
        src_pad_mask = self.get_pad_mask(src) if hasattr(self.args, 'pad_token') else None

        # Embedding and positional encoding.
        embedded = self.enc_e(src)  # (batch, seq_len, embd_dim)

        # Transformer encoder and MLP.
        memory = self.encoder0(embedded, src_key_padding_mask=src_pad_mask)  # (batch, seq_len, embd_dim)
        memory = self.module0(memory)  # (batch, seq_len, embd_dim)

        # Create a mask for valid (non-pad) tokens
        if src_pad_mask is not None:
            src_pad_mask = ~src_pad_mask  # Invert mask (True for non-pad tokens)
            src_pad_mask = src_pad_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            print(src_pad_mask[0][0])
            exit()
            # Apply mask before mean pooling
            memory_masked = memory * src_pad_mask  # Mask out pad embeddings
            sum_embeddings = memory_masked.sum(dim=1)  # Sum valid embeddings
            valid_token_counts = src_pad_mask.sum(dim=1)  # Count valid tokens
            embedding = sum_embeddings / valid_token_counts  # Compute mean only over non-pad tokens
        else:
            embedding = memory.mean(dim=1)  # Fallback to regular mean pooling if no padding mask
        
        return embedding  # (batch, embd_dim)

    def forward(self, src1, src2):
        """
        Forward pass for two sequences.
        
        Args:
            src1: Tensor of shape (batch, seq_len)
            src2: Tensor of shape (batch, seq_len)
        
        Returns:
            emb1: Embedding for src1, Tensor of shape (batch, embd_dim)
            emb2: Embedding for src2, Tensor of shape (batch, embd_dim)
        """
        emb1 = self._process(src1)
        emb2 = self._process(src2)
        return emb1, emb2