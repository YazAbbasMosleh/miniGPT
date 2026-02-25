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
    A more advanced self-attention mechanism implementation that includes
    multi-head attention and batch processing capabilities.
    
    This module implements multi-head self-attention where input sequences
    attend to themselves across multiple representation subspaces. It projects
    input vectors into query, key, and value spaces using learnable weight matrices,
    computes attention scores for each head, and concatenates the resulting context
    vectors before a final linear transformation.
    
    Args:
        d_in (int): Dimension of input features (embedding dimension)
        d_out (int): Dimension of output features (attention output dimension)
        qkv_bais (int): Number of attention heads (must divide d_out evenly)
    returns:
        torch.Tensor: Context vectors of shape (batch_size, seq_len, d_out)
            Each output vector contains information aggregated from all
            positions in the input sequence, weighted by attention scores
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
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape(-1)**0.5, dim=-1)
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