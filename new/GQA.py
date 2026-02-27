import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    batch_size, seq_len, q_dim = Q.shape
    head_dim = q_dim // num_heads
    num_groups = num_heads // num_kv_heads
    
    # Reshape to separate heads: (batch, seq, heads, head_dim)
    # Then transpose to: (batch, heads, seq, head_dim)
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Expand K, V by repeating each kv head for its group of query heads
    K = np.repeat(K, num_groups, axis=1)
    V = np.repeat(V, num_groups, axis=1)
    
    # Scaled dot-product attention
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V)
    
    # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, heads * head_dim)
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, num_heads * head_dim)
    
    return output