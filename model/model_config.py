from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """
    Model configurations parameters 

    Args:
        dim (int): Frequency tensor to be reshaped.
        n_layers (int): Number of layers in model
        n_heads (int): Number of heads in self-attention module 
        n_kv_heads Optional[int]: Number of key and value heads in module (default None)
        vocab_size (int): Number of tokens in vocabulary
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value
        ffn_dim_multiplier Optional[float]: Custom multiplier for hidden dimension. (default is None)
        norm_eps (float): A small number to RMSNorm denominator for numerical stability
    Returns:
        [int], [float] : model config parameters

    """

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048