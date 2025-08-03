
def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation

    example:
    假设 head_dim = 128, base = 10000
    dim = 128 (全部维度参与旋转)
    torch.arange(0, 128, 2) = [0, 2, 4, ..., 126]
    # 对于维度0：
    inv_freq[0] = 1.0 / (10000^(0/128)) = 1.0
    # 对于维度2：
    inv_freq[1] = 1.0 / (10000^(2/128)) ≈ 0.95
    # 对于维度126：
    inv_freq[63] = 1.0 / (10000^(126/128)) ≈ 0.0001
    """
    # Compute the inverse frequencies
    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor
