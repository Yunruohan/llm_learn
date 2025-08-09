import torch
from torch import nn
import torch.nn.functional as F


# 假设的形状
batch_size = 2
seq_length = 5
hidden_dim = 3
num_experts = 2

# 初始化 hidden_states
hidden_states = torch.tensor([
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
    [[1.6, 1.7, 1.8], [1.9, 2.0, 2.1], [2.2, 2.3, 2.4], [2.5, 2.6, 2.7], [2.8, 2.9, 3.0]]
], dtype=torch.float32)  # 形状为 [batch_size, seq_length, hidden_dim]

# 初始化 expert_mask
expert_mask = torch.tensor([
    [[1, 0], [0, 1], [0, 0], [1, 0], [0, 1]],
    [[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]]
], dtype=torch.float32)  # 形状为 [num_experts, seq_length, batch_size]

# 初始化 routing_weights
routing_weights = torch.tensor([
    [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.8, 0.2], [0.3, 0.7]],
    [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]
], dtype=torch.float32)  # 形状为 [num_experts, seq_length, batch_size]

# 初始化 final_hidden_states
final_hidden_states = torch.zeros(seq_length, batch_size, hidden_dim, dtype=torch.float32)


class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 假设的专家层
class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

        # 多专家
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        # 专家门控
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 共享专家
        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        # 共享专家门控
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        # 手动定义权重和偏置
        self.linear.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        self.linear.bias = torch.nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 将输入的隐藏状态展平为二维张量 [batch_size * sequence_length, hidden_dim]
        hidden_states = hidden_states.view(-1, hidden_dim)
        print('hidden_states: ', hidden_states.shape)
        # 路由逻辑：计算每个 token 的路由逻辑值，形状为 (batch_size * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        print('router_logits: ', router_logits.shape)
        # 对路由逻辑值应用 Softmax 函数，得到每个 token 分配给每个专家的权重, [batch_size * sequence_length, n_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        print('routing_weights: ', routing_weights.shape)
        # print(routing_weights)
        # 选择每个 token 的 top-K 个专家，并获取对应的权重和专家索引, [batch_size * sequence_length, n_experts]
        # routing_weights: 每个 token 选择的 top-K专家权重矩阵，相当于routing_weights第二维按专家权重排序，取 top-K
        # selected_experts: 上面权重对应的专家 index，即每个token都选择 top-K个权重最大的专家，这里用来记录选择的专家编号
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states.dtype)
        print(routing_weights.shape, selected_experts.shape)
        # print(routing_weights)
        # print(selected_experts)
        # 初始化一个零张量，用于存储最终的输出, [batch_size * sequence_length, hidden_dim]
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # 专家掩码：标记哪些 token 被分配给哪些专家，[n_experts, top-k, batch_size * sequence_lengt，        
        # 本质是标记 哪些 token 被分配给哪些专家，只是成了专家->token维度
        # 先使用one_hot，构建专家索引矩阵，记录每个 token 选择每个专家的one-hot，[batch_size * sequence_lengt, top-k, n_experts]
        # 使用permute函数进行维度调整，[n_experts, top-k, batch_size * sequence_lengt], 表示每个专家被选中，要生效的 token，
        # 本质是标记 哪些 token 被分配给哪些专家，只是成了专家->token维度
        print(torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).shape)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        print(expert_mask.shape)
        # 按-1，-2 维度 sum，并且大于 0 的，表示有被 token 选择的专家
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        print(expert_hitted.shape)
        # 遍历有被选中的专家，计算专家输出值
        for expert_idx in expert_hitted:
            # 取专家网络
            expert_layer = self.experts[expert_idx]
            # 专家被选择 top-K索引，以及生效的 token，idx表示 topk索引，top_x表示 token索引
            # torch.where等价于torch.nonzero，即返回非 0 索引
            # expert_mask[expert_idx]形状是[top-K, batch_size * sequence_lengt], 表示expert_idx专家作为 top-k 选择时，对哪些 token生效
            # torch.nonzero返回 top-k和batch_size * sequence_lengt维度 非 0 值的索引
            # idx outjoin top_x 即对应 expert_mask[expert_idx]非 0 值二维索引
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            print(expert_mask[expert_idx].squeeze(0))
            print('idx: ', idx)
            print('top_x: ', top_x)
            print(hidden_states[None, top_x].shape)
            # 取生效的 token 隐层输入向量，[专家生效 token数量, hidden_dim]
            # 等价于 hidden_states[top_x].reshape(-1, hidden_dim)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            # 专家输出向量：生效的输入向量经专家层得到专家编码，再乘以专家权重，得到专家输出向量, [专家生效 token 数量， hidden_dim]
            # routing_weights[top_x, idx, None]最后一维加 None是增加维度，routing_weights[top_x, idx]这样是一维列表
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            print(current_hidden_states.shape)
            print(expert_idx)
            # token输出向量累加生效专家输出：在final_hidden_states第 0 维度为top_x索引位置累加current_hidden_states
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        print(final_hidden_states)
        # 共享专家输出向量：共享专家层输出乘以共享专家门控（或权重），[batch_size * sequence_length, hidden_dim]
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        # 最后输出向量：多专家输出(final_hidden_states) + 共享专家输出(shared_expert_output)
        final_hidden_states = final_hidden_states + shared_expert_output
        
        # 恢复输入 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states



class MoeConfig(object):
    def __init__(self):
        self.hidden_size = 3
        self.num_experts = 2
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 100
        self.shared_expert_intermediate_size = 100

config = MoeConfig()
export_layer = ExpertLayer(config)

export_layer(hidden_states)

# experts = [ExpertLayer(hidden_dim) for _ in range(num_experts)]

# batch_size, sequence_length, hidden_dim = hidden_states.shape
# # 将输入的隐藏状态展平为二维张量 [batch_size * sequence_length, hidden_dim]
# hidden_states = hidden_states.view(-1, hidden_dim)


# # 假设 expert_hitted
# expert_hitted = torch.tensor([[0], [1]])
# for expert_idx in expert_hitted:
#     expert_layer = experts[expert_idx.item()]
#     idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
#     print(idx, top_x)
#     # 提取当前专家被选中的隐藏状态
#     # 
#     current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#     print(current_state.shape)
#     print(expert_idx)
#     print(expert_layer(current_state))
#     print(routing_weights.shape)
#     current_hidden_states = expert_layer(current_state) * routing_weights[expert_idx.item(), top_x, idx]
#     print(routing_weights[expert_idx.item(), top_x, idx, None])
#     break