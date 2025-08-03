import torch
from torch import nn
from typing import Callable, Optional, Union

from configuration_llama import LlamaConfig

from utils.cache_utils import DynamicCache
from utils.masking_utils import create_causal_mask
from utils.modeling_rope_utils import LlamaRotaryEmbedding

from modeling_outputs import BaseModelOutputWithPast


class LlamaRMSNorm(nn.Module):
    """
    y = γ * (x / √(mean(x²) + ε))
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习的缩放参数，初始化为全1向量
        self.variance_epsilon = eps # 数值稳定性常数，防止除零错误


    def forward(self, hidden_states):
        """
        hidden_states: 隐藏输出状态
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 对每个元素计算平方，在最后一个维度（特征维度）上计算均值
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        # SwiGLU(x) = SiLU(Linear(x)) * Linear(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.

    Example:
        input: [x0, x1, x2, x3, x4, x5]
        output: [-x3, -x4, -x5, x0, x1, x2] 
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    """
    # 扩展 cos 和 sin 维度, [batch_size, seq_len, head_dim] -> [batch_size, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # 应用旋转公式, 这里不是连续位置(x0, x1)共享旋转角度, 而是(x0, x_{dim//2}), (x1, x_{dim//2 + 1} ...) 这样分组共享旋转角度
    # x_0_new = x0 * cos - x_{dim//2} * sin, x_{dim//2}_new = x_{dim//2} * cos + x0 * sin
    # x_1_new = x1 * cos - x_{dim//2 + 1} * sin, x_{dim//2 + 1}_new = x_{dim//2 + 1} * cos + x1 * sin
    # ...
    # 优点：交错设计提供了更好的频率分布与丰富的位置编码模式, 实验证明效果更好
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

   
def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    """
    hidden_states: [batch_size, seq_len, num_key_value_heads, head_dim]
    n_rep: group_num
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_attention_heads, seq_len, head_dim]
    # 其中num_attention_heads = n_rep * num_kv_heads
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    # **kwargs: Unpack[TransformersKwargs],
):
    """
    attention 
    attention_mask: 4D [batch_size, head_num, seq_length, kv_length]
    """
    key_states = repeat_kv(key, module.num_key_value_groups) # [batch_size, num_head, seq_len, head_dim]
    value_states = repeat_kv(value, module.num_key_value_groups) # [batch_size, num_head, seq_len, head_dim]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # [batch_size, num_head, seq_len, seq_len]
    
    # 掩码，避免信息泄露
    if attention_mask is not None:
        # 掩码切片，位置 i 只能看到位置 0 到 i 的信息，key_states.shape[-2]是当前序列的长度
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # 掩码中的 -inf 值会使对应位置的注意力权重变为 -inf, 掩码中的 0 值不会影响注意力权重
        attn_weights = attn_weights + causal_mask

    # [batch_size, head_num, seq_len, seq_len]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states) # [batch_size, num_head, seq_len, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous() # [batch_size, seq_len, num_head, head_dim]
    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """
    GQA: 
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        # Decoder层数
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        # kv group
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # scaling作用？
        self.scaling = self.head_dim**-0.5

        # attention_bias 是什么？
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias
        )
        #  key 和 value只使用 num_key_value_heads 个头
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,

    ) -> tuple[torch.Tensor, torch.Tensor]:
        print(len(position_embeddings))
        """
        hidden_state: [batch_size, seq_len, hidden_size]
        """
        input_shape = hidden_states.shape[:-1] # [batch_size, seq_len]
        hidden_shape = [*input_shape, -1, self.head_dim] #[batch_size, seq_len, -1, head_dim]

        # hidden_states -> query、key and value 
        # step1: q_proj(hidden_states), [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_attention_heads * head_dim]
        # step2: view(hidden_shape), -> [batch_size, seq_len, num_attention_heads, head_dim]
        # step3: transpose(1,2), -> [batch_size, num_attention_heads, seq_len, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [batch_size, num_key_value_heads, seq_len, head_dim]
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 应用位置编码
        # 每次计算 attention时使用, pos_embeding是固定的, 不会参与梯度更新
        cos, sin = position_embeddings
        print('******** layer id: <{}> positiont_embeddings ********'.format(self.layer_idx))
        print('cos: ', cos.shape)
        print('sin: ', sin.shape)
        query_states, key_status = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 缓存
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention计算
        # [batch_size, seq_len, num_head, head_dim]
        attn_output, attn_weights = eager_attention_forward(
            self, 
            query_states,
            key_states,
            value_states,
            attention_mask,
            # dropout=0.0 if not self.training else self.attention_dropout,
            # dropout= self.attention_dropout,
            scaling=self.scaling,
        )
        # 恢复输入shape: [batch_size, seq_len, hidden_size]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # MLP
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        # 隐层维度
        self.hidden_size = config.hidden_size
        # attention模块
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        # MLP
        self.mlp = LlamaMLP(config)
        # 输入 Norm
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # attention后的 Norm
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        # **kwargs: Unpack[TransformersKwargs],
    ):
        print('Decoder {}:'.format(self.layer_idx))
        print(hidden_states.shape)
        residual = hidden_states
        # 对输入向量进行 Norm
        hidden_states = self.input_layernorm(hidden_states)
        print('layer_norm: ', hidden_states.shape)
        # 注意力计算
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            past_key_value=past_key_value,
            # use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # 残差
        hidden_states = residual + hidden_states

        # 全连接层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        print('final output: ', hidden_states.shape)
        return hidden_states


class LlamaPreTrainedModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        pass


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 词表维度[vocab_size, hidden_size], padding_idx会初始化 0，不参与梯度更新
        print(config.vocab_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 解码层
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # # Norm层
        # self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 旋转位置编码层
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # 暂不实现
        # self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # 异或操作：保证输入要么是 input_ids, 要么是inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # embedding化
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        # 初始化K-V缓存
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # cache_position: 当前输入序列中每个 token在完整序列中的绝对位置, 形状(query_length, )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        # 每个 token在序列中的位置编码, 形状[batch_size, query_lenght]
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # 创建4D掩码向量, [batch_size, 1, query_lenght, kv_length]
        # query_length: 当前 输入序列的索引
        # kv_length: attention计算时需要的 key-value 大小
        casual_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        hidden_states = inputs_embeds
        # 计算位置编码
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=casual_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings
                # **kwargs,
            )
        
        # hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )


def _test_rmsnorm():
    x = torch.randn(3, 4, 5) # [batch_size, seq_lenght, dim]
    print('张量x:', x)

    # 创建 RMSNorm 实例
    rms_norm = LlamaRMSNorm(5)
    # 前向传播
    output = rms_norm(x)
    print('\nRMSNorm 归一化后的输出:')
    print(output)


def _test_attention():
    config = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8
    )
    batch_size = 2
    seq_len = 3
    num_head = 4
    head_dim = 10
    query = torch.randn(batch_size, seq_len, num_head, head_dim)
    key = torch.randn(batch_size, seq_len, num_head, head_dim)
    val = torch.randn(batch_size, seq_len, num_head, head_dim)
    hidden_states = torch.randn(batch_size, seq_len, 32)

    attention = LlamaAttention(config=config)
    out, _ = attention(hidden_states)
    # out = eager_attention_forward(None, query, key, val, scaling=1.0)
    
    print(out.shape)


def _test_llama_model():
    config = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8
    )
    llama_model = LlamaModel(config)
    llama_model(
        input_ids=torch.LongTensor([[1,2,3,4,5]])
    )

def main():
    # _test_rmsnorm()
    # _test_attention()
    _test_llama_model()

if __name__ == '__main__':
    main()