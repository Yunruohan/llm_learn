import torch
from typing import Callable, Optional, Union
from typing import Any, Optional, Union


class DynamicCache(object):
    # 仅在推理阶段使用？

    def __init__(self):
        # key_cache[layer_idx].shape = (batch_size, num_heads, seq_len, head_dim)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Return:
            A tuple containing the updated key and value states.
        """
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # 当缓存 layer层数小于索引层id时，需要将缓存层到索引层之间填充占位符
                # 
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(Tensor([]))
                    self.value_cache.append(Tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(val_states)
            elif (not self.key_cache[layer_idx].numel()):
                # 当前层索引值为空，填充 key/value states
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = val_states
            else:
                # 当前层索引已有值，则将 key/value states与缓存值拼接，
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], val_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


