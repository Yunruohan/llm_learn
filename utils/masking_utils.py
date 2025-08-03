from typing import Callable, Optional, Union

import torch

from utils.cache_utils import DynamicCache

BlockMask = torch.Tensor


class PretrainedConfig(object):
    pass


def and_masks(batch_idx, head_idx, q_idx, kv_idx):
    result = q_idx.new_ones((), dtype=torch.bool)
    
    result = result & mask(batch_idx, head_idx, q_idx, kv_idx)

    return result


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    
    return mask_function


def sdpa_mask_recent_torch(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[torch.Tensor] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.
    """
    q_length = cache_position.shape[0]
    
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    # 输入是四个张量，按照不同维度比较大小，返回 4d张量
    # batch_indices = torch.tensor([0, 1])      # 批次索引
    # head_indices = torch.tensor([0, 1])       # 头索引
    # q_indices = torch.tensor([0, 1, 2])       # 查询索引
    # kv_indices = torch.tensor([0, 1, 2])      # 键值索引
    # 输出: 4D张量 (2, 2, 3, 3)
    causal_mask = _vmap_for_bhqkv(causal_mask_function)(batch_arange, head_arange, cache_position, kv_arange)
    return causal_mask


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[DynamicCache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an HybridCache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    """
    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    layer_idx = 0
    # If using a cache, it can give all informations about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0
    
    mask_factory_function = causal_mask_function

    causal_mask = sdpa_mask_recent_torch(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=True,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask