import torch
import torch.nn as nn
from attention.self_attention import CasualSelfAttention



class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in: int,
                 d_out: int,
                 context_length: int,
                 dropout: float,
                 num_heads: int | None = None,
                 qkv_bias: bool=False,
                 automatic_num_heads: bool=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout = dropout

        if automatic_num_heads and num_heads is not None:
            raise ValueError("Cannot specify num_heads when automatic_num_heads is True.")
        if not automatic_num_heads and num_heads is None:
            raise ValueError("Must specify num_heads when automatic_num_heads is False.")
        
        if automatic_num_heads:
            num_heads = max(1, d_out // 64)
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads
        
        self.heads = nn.ModuleList([
            CasualSelfAttention(d_in=d_in, d_out=self.head_dim, context_length=context_length, dropout=dropout, qkv_bias=qkv_bias)
            for _ in range(self.num_heads)
        ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module as described in "Attention Is All You Need".

    This module computes scaled dot-product attention in parallel across multiple heads,
    allowing the model to jointly attend to information from different representation subspaces.

    Attributes:
        d_in (int): Input feature dimension.
        d_out (int): Output feature dimension.
        num_heads (int): Number of attention heads. Must divide `d_out` evenly.
        head_dim (int): Dimension of each head, computed as `d_out // num_heads`.
        W_key (nn.Linear): Linear projection for keys. In_features=d_in, out_features=d_out.
        W_query (nn.Linear): Linear projection for queries. In_features=d_in, out_features=d_out.
        W_value (nn.Linear): Linear projection for values. In_features=d_in, out_features=d_out.
        out_proj (nn.Linear): Final linear projection. In_features=d_out, out_features=d_out.
        dropout (nn.Dropout): Dropout layer applied to attention weights. Default probability = 0.0.
        mask (torch.Tensor, optional): Causal / attention mask of shape (max_seq_len, max_seq_len).
            Typically a triangular matrix to prevent attending to future tokens. Defaults to None.

    Note:
        The code currently contains a bug: after reshaping `keys`, the lines
        `values = keys.view(...)` and `queries = keys.view(...)` mistakenly reuse the
        `keys` tensor instead of the correctly projected `values` and `queries` tensors.
        The intended behavior is to reshape each projected tensor separately.
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 dropout: float, num_heads: int, qkv_bias: bool=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        
        self.head_dim = self.d_out // self.num_heads
        
        self.W_query = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=self.qkv_bias)
        self.W_key = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=self.qkv_bias)
        self.W_value = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=self.qkv_bias)
        
        self.out_proj = nn.Linear(in_features=self.d_in, out_features=self.d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length)), diagonal=1)
    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out).

        Shape:
            - Input: (b, T, d_in) where b = batch size, T = sequence length, d_in = input dim.
            - Intermediate:
                * keys/ queries/ values after projection: (b, T, d_out)
                * Reshaped for multi-head: (b, T, num_heads, head_dim)
                * Transposed: (b, num_heads, T, head_dim)
                * Attention scores: (b, num_heads, T, T)
                * Attention weights: (b, num_heads, T, T)
                * Context vectors before projection: (b, num_heads, T, head_dim) -> (b, T, d_out)
            - Output: (b, T, d_out)

        Variables:
            b (int): Batch size.
            num_tokens (int): Sequence length (T).
            d_in (int): Input feature dimension.
            keys, queries, values (torch.Tensor): Projected key, query, value tensors.
            attn_scores (torch.Tensor): Raw attention scores before masking.
            mask_bool (torch.Tensor): Boolean mask extracted from `self.mask`.
            attn_weights (torch.Tensor): Attention probabilities after softmax and dropout.
            context_vec (torch.Tensor): Weighted sum of values, then reshaped and projected.

        Examples:
            >>> # Create a multi-head attention layer
            >>> d_in, d_out, num_heads = 512, 512, 8
            >>> mha = MultiHeadAttention(d_in, d_out, num_heads, dropout=0.1)
            >>> # Random input: batch=2, sequence length=10, dim=512
            >>> x = torch.randn(2, 10, d_in)
            >>> output = mha(x)
            >>> output.shape
            torch.Size([2, 10, 512])

        Note:
            The current implementation has a bug: after reshaping `keys`,
            the tensors `values` and `queries` are incorrectly assigned from
            the reshaped `keys` instead of from their own projections. This
            causes all three to be identical, breaking the attention mechanism.
            The intended code should read:
                keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
                values = values.view(b, num_tokens, self.num_heads, self.head_dim)
                queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        """
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attn_scores = attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec