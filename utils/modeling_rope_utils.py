import torch
from torch import nn
from typing import Callable, Optional, Union

from models.llama.configuration_llama import LlamaConfig


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度, 即theta_{j} = 10000^{(-1 * j)/d}, 其中，j in (0, ..., d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    # 生成 token 序列索引 [0, 1, ..., seq_len-1] 
    t = torch.arange(seq_len, device=freqs.device)
    # [seq_len, dim/2], 这里序列 idx也会乘以旋转角度, 从行看，单调递减，从列看，单调递减
    freqs = torch.outer(t, freqs).float()
    print(freqs)
    # torch.polar(abs, angle, *, out=None) -> out=abs⋅cos(angle)+abs⋅sin(angle)⋅j
    # 求每个位置对应 theta的 cos + sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    print(freqs_cis.shape)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq_shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk_shape[:-1], -1, 2)

    # [batch_size, seq_len, dim/2], 每个元素是复数
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def _compute_default_rope_parameters(
    config: Optional[LlamaConfig] = None,
    device: Optional["torch.device"] = None
    ):
    # theta
    base = config.rope_theta
    
    head_dim = config.hidden_size // config.num_attention_heads
    dim = head_dim
    
    attention_factor = 1.0  # Unused in this type of RoPE

    # 计算逆频率
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)/dim))
    return inv_freq, attention_factor


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.config = config
        # 计算逆频率
        self.rope_init_fn = _compute_default_rope_parameters
        # inv_freq: [head_dim//2]
        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # 扩展逆频率维度 [dim//2] -> [batch_size, head_dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # 扩展位置编码维度 [batch_size, seq_len] -> [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        # 
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # 计算频率 [batch_size, seq_len, head_dim//2]
            # 等价于 torch.outer(position_ids_expanded.float().transpose(1, 2), inv_freq_expanded.float().transpose(1, 2))
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # 复制频率到完整维度 [batch_size, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # 计算 cos 和 sin
            cos = emb.cos() 
            sin = emb.sin()
            print('cos: ', cos.shape)
            print('sin: ', sin.shape)
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def main():
    dim = 10
    seq_len = 3
    precompute_freqs_cis(dim, seq_len)


# if __name__ == '__main__':
#     main()

# inv_freq = [dim/2] -> batch, dim/2, 1