
class LlamaConfig(object):
    def __init__(
        self,
        vocab_size=32000,              # 词表大小
        hidden_size=4096,              # 隐藏层维度
        intermediate_size=11008,       # MLP中间层维度 
        num_hidden_layers=32,          # Transformer层数
        num_attention_heads=32,        # 注意力头数
        num_key_value_heads=None,      # GQA的Key/Value头数
        head_dim=None,
        attention_bias=False,          # 是否使用注意力偏置
        rms_norm_eps=1e-6,
        pad_token_id=None,             # pad token ID
        attention_dropout=0.0,
        mlp_bias=False,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.pad_token_id = pad_token_id
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta


def my_test_funciton():
    pass