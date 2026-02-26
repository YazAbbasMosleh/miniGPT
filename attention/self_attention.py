import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    """
    A basic self-attention mechanism implementation without batch processing.
    
    This module implements scaled dot-product self-attention where input sequences
    attend to themselves. It projects input vectors into query, key, and value spaces
    using learnable weight matrices, then computes attention scores and weighted
    context vectors.
    
    Args:
        d_in (int): Dimension of input features (embedding dimension)
        d_out (int): Dimension of output features (attention output dimension)
    
    Attributes:
        W_query (nn.Parameter): Learnable weight matrix for query projection
            Shape: (d_in, d_out)
        W_key (nn.Parameter): Learnable weight matrix for key projection
            Shape: (d_in, d_out)
        W_value (nn.Parameter): Learnable weight matrix for value projection
            Shape: (d_in, d_out)
    
    Shape:
        - Input: (seq_len, d_in) where seq_len is the sequence length
        - Output: (seq_len, d_out) where each position contains the
          context-aggregated representation
    
    Example:
        >>> d_in, d_out = 512, 256
        >>> seq_len = 10
        >>> x = torch.randn(seq_len, d_in)
        >>> attention = SelfAttention_v1(d_in, d_out)
        >>> context = attention(x)
        >>> print(context.shape)
        torch.Size([10, 256])
    
    Note:
        This implementation assumes single-head attention without batch processing.
        For batched inputs, the forward method would need modification to handle
        batch dimensions properly.
        
    Theory:
        The self-attention mechanism computes:
        1. Query, Key, Value projections: Q = x @ W_query, K = x @ W_key, V = x @ W_value
        2. Attention scores: scores = Q @ K^T
        3. Scaled attention weights: weights = softmax(scores / sqrt(d_out))
        4. Context vectors: context = weights @ V
        
        The scaling factor 1/sqrt(d_out) prevents dot products from growing too large
        in magnitude, which would push the softmax function into regions of extremely
        small gradients.
    """
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        """
        Compute self-attention for input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, d_in)
                where seq_len is the number of tokens/positions in the sequence
        
        Returns:
            torch.Tensor: Context vectors of shape (seq_len, d_out)
                Each output vector contains information aggregated from all
                positions in the input sequence, weighted by attention scores
        
        Process:
            1. Project input to query, key, and value spaces
            2. Compute attention scores by matching queries with keys
            3. Apply softmax with scaling to get attention weights
            4. Aggregate values using attention weights to produce context
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec





class SelfAttention_v2(nn.Module):
    """
    Single-head scaled dot-product self-attention layer.

    This module implements the standard self-attention mechanism without
    causal masking. Each token in the input sequence is allowed to attend
    to all other tokens (including future tokens).

    The attention computation follows:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    where Q, K, and V are linear projections of the input.

    Parameters
    ----------
    d_in : int
        Input embedding dimension.

    d_out : int
        Output embedding dimension (dimension of projected queries,
        keys, and values).

    qkv_bias : bool, optional (default=False)
        If True, adds a learnable bias term to the query, key, and value
        linear projections.

    Attributes
    ----------
    W_query : torch.nn.Linear
        Linear projection layer that maps input embeddings to query vectors.

    W_key : torch.nn.Linear
        Linear projection layer that maps input embeddings to key vectors.

    W_value : torch.nn.Linear
        Linear projection layer that maps input embeddings to value vectors.

    Notes
    -----
    - This implementation does NOT apply causal masking.
    - Every token can attend to every other token.
    - This layer is typically used in encoder-style transformers
      (e.g., BERT), where bidirectional attention is allowed.
    - For autoregressive models (e.g., GPT), a causal mask must be added.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> batch_size = 2
    >>> seq_len = 4
    >>> d_model = 8
    >>>
    >>> x = torch.randn(batch_size, seq_len, d_model)
    >>>
    >>> attn = SelfAttention_v2(
    ...     d_in=d_model,
    ...     d_out=d_model,
    ...     qkv_bias=False
    ... )
    >>>
    >>> output = attn(x)
    >>> output.shape
    torch.Size([2, 4, 8])
    """
    def __init__(self, d_in, d_out, qkv_bais=False):
        super().__init__()
        self.d_int = d_in
        self.d_out = d_out
        self.qkv_bais = qkv_bais
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bais)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bais)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bais)
    def forward(self, x):
        """
        Perform the forward pass of self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        torch.Tensor
            Contextualized output tensor of shape
            (batch_size, num_tokens, d_out).

        Description
        -----------
        1. Project the input tensor into queries, keys, and values.
        2. Compute attention scores using scaled dot-product.
        3. Apply softmax over the last dimension to obtain attention weights.
        4. Compute the weighted sum of value vectors.

        Shape Details
        -------------
        queries : (batch_size, num_tokens, d_out)
        keys    : (batch_size, num_tokens, d_out)
        values  : (batch_size, num_tokens, d_out)

        attn_scores : (batch_size, num_tokens, num_tokens)
        attn_weights: (batch_size, num_tokens, num_tokens)
        context_vec : (batch_size, num_tokens, d_out)
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape(-1)**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec





class CasualSelfAttention:
    """
    Causal (masked) self-attention layer.

    This module implements single-head causal self-attention as used in
    autoregressive transformer models (e.g., GPT-style models). It prevents
    each token from attending to future tokens by applying an upper-triangular
    mask to the attention score matrix before softmax.

    The attention mechanism computes:

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    where future positions are masked with -inf to enforce causality.

    Parameters
    ----------
    d_in : int
        Input embedding dimension.

    d_out : int
        Output embedding dimension (dimension of query, key, and value projections).

    context_length : int
        Maximum sequence length supported by the model. This determines
        the size of the causal mask.

    dropout : float
        Dropout probability applied to the attention weights.
        Must be in the range [0, 1].

    qkv_bias : bool, optional (default=False)
        If True, adds a learnable bias to the query, key, and value
        linear projections.

    Attributes
    ----------
    W_query : torch.nn.Linear
        Linear projection layer for queries.

    W_key : torch.nn.Linear
        Linear projection layer for keys.

    W_value : torch.nn.Linear
        Linear projection layer for values.

    dropout : torch.nn.Dropout
        Dropout layer applied to attention weights.

    mask : torch.Tensor
        Upper-triangular causal mask of shape (context_length, context_length).
        Registered as a buffer (non-trainable tensor that moves with the model).

    Notes
    -----
    - The causal mask ensures that token at position i cannot attend to
      any position j > i.
    - The mask is registered as a buffer so it:
        * Moves automatically with the model to CPU/GPU.
        * Is saved in the model state_dict.
        * Does not receive gradients.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>> 
    >>> batch_size = 2
    >>> seq_len = 4
    >>> d_model = 8
    >>> 
    >>> x = torch.randn(batch_size, seq_len, d_model)
    >>> 
    >>> attn = CasualSelfAttention(
    ...     d_in=d_model,
    ...     d_out=d_model,
    ...     context_length=seq_len,
    ...     dropout=0.1,
    ...     qkv_bias=False
    ... )
    >>> 
    >>> output = attn(x)
    >>> output.shape
    torch.Size([2, 4, 8])
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bais=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.W_query = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bais)
        self.W_key = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bais)
        self.W_value = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bais)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )
    
    def forward(self, x):
        """
        Perform the forward pass of causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_tokens, d_in).

        Returns
        -------
        torch.Tensor
            Contextualized output tensor of shape
            (batch_size, num_tokens, d_out).

        Description
        -----------
        1. Project input into queries, keys, and values.
        2. Compute scaled dot-product attention scores.
        3. Apply causal mask to prevent attending to future tokens.
        4. Apply softmax to obtain attention weights.
        5. Apply dropout to attention weights.
        6. Compute weighted sum of value vectors.

        Shape Details
        -------------
        queries : (batch_size, num_tokens, d_out)
        keys    : (batch_size, num_tokens, d_out)
        values  : (batch_size, num_tokens, d_out)

        attn_scores : (batch_size, num_tokens, num_tokens)
        attn_weights: (batch_size, num_tokens, num_tokens)
        context_vec : (batch_size, num_tokens, d_out)
        """
        b , num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape(-1)**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec






if __name__ == "__main__":
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]] # step
        )
    
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(3, 2)
    print(sa_v1(inputs))