import torch
import torch.nn as nn
from typing import Dict
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.generation.utils import GenerationConfig
import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import gc

import traceback
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from transformers.cache_utils import DynamicCache

import flashinfer

class PredictorDynamicCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.predictor_primary_key: List[Optional[torch.Tensor]] = []
        self.predictor_primary_value: List[Optional[torch.Tensor]] = []
        self.predictor_importance_key: List[Optional[torch.Tensor]] = []

    def update_predictor_primary(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append or create the predictor's "primary" K/V states for `layer_idx`.

        shape for key_states, value_states is typically [batch_size, num_heads, seq_len, head_dim].
        """
        # Extend the lists so that `predictor_primary_key[layer_idx]` and
        # `predictor_primary_value[layer_idx]` exist.
        self._ensure_list_capacity(
            self.predictor_primary_key, layer_idx, fill=None
        )
        self._ensure_list_capacity(
            self.predictor_primary_value, layer_idx, fill=None
        )

        # If this is the very first time we are updating that layer's predictor cache, just assign
        if self.predictor_primary_key[layer_idx] is None:
            self.predictor_primary_key[layer_idx] = key_states
            self.predictor_primary_value[layer_idx] = value_states
        else:
            # Otherwise, concatenate along the seq_len dimension (=-2 or =2 depending on your shape).
            self.predictor_primary_key[layer_idx] = torch.cat(
                [self.predictor_primary_key[layer_idx], key_states], dim=2
            )
            self.predictor_primary_value[layer_idx] = torch.cat(
                [self.predictor_primary_value[layer_idx], value_states], dim=2
            )

        return (
            self.predictor_primary_key[layer_idx],
            self.predictor_primary_value[layer_idx],
        )

    def update_predictor_importance(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Append or create the predictor's "importance" key for `layer_idx`.
        """
        self._ensure_list_capacity(
            self.predictor_importance_key, layer_idx, fill=None
        )

        if self.predictor_importance_key[layer_idx] is None:
            self.predictor_importance_key[layer_idx] = key_states
        else:
            self.predictor_importance_key[layer_idx] = torch.cat(
                [self.predictor_importance_key[layer_idx], key_states], dim=2
            )
        return self.predictor_importance_key[layer_idx]

    def crop(self, max_length: int):
        super().crop(max_length)
        # Now also crop predictor caches
        for idx in range(len(self.predictor_primary_key)):
            if self.predictor_primary_key[idx] is not None:
                self.predictor_primary_key[idx] = self.predictor_primary_key[idx][..., :max_length, :]
                self.predictor_primary_value[idx] = self.predictor_primary_value[idx][..., :max_length, :]

        for idx in range(len(self.predictor_importance_key)):
            if self.predictor_importance_key[idx] is not None:
                self.predictor_importance_key[idx] = self.predictor_importance_key[idx][..., :max_length, :]

        # Remember to adjust self._seen_tokens accordingly
        self._seen_tokens = min(self._seen_tokens, max_length)

    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["PredictorDynamicCache"]:
        # Use the base split logic for the standard K/V
        base_splits = super().batch_split(full_batch_size, split_size, num_hidden_layers)
        # `base_splits` is now a list of new DynamicCache objects. But we *actually*
        # want them to be PredictorDynamicCache so we can store the predictor states.
        # Easiest: we can cast and fill them. 
        out: List[PredictorDynamicCache] = []

        for split_i, base_split in enumerate(base_splits):
            # Construct an empty PredictorDynamicCache
            new_cache = PredictorDynamicCache()
            # Copy over the underlying fields from base_split
            new_cache.key_cache = base_split.key_cache
            new_cache.value_cache = base_split.value_cache
            new_cache._seen_tokens = base_split._seen_tokens

            # Now also slice our predictor fields
            # The slice in batch dim is [i:i+split_size].
            b_start = split_i * split_size
            b_end = min(full_batch_size, b_start + split_size)

            new_cache.predictor_primary_key = self._slice_list_tensors(
                self.predictor_primary_key, b_start, b_end
            )
            new_cache.predictor_primary_value = self._slice_list_tensors(
                self.predictor_primary_value, b_start, b_end
            )
            new_cache.predictor_importance_key = self._slice_list_tensors(
                self.predictor_importance_key, b_start, b_end
            )

            out.append(new_cache)

        return out

    @classmethod
    def from_batch_splits(cls, splits: List["PredictorDynamicCache"], num_hidden_layers: int = None) -> "PredictorDynamicCache":
        # Let the base class handle the normal K/V merges
        base_merged = DynamicCache.from_batch_splits(splits, num_hidden_layers=num_hidden_layers)
        merged = cls()
        merged.key_cache = base_merged.key_cache
        merged.value_cache = base_merged.value_cache
        merged._seen_tokens = base_merged._seen_tokens

        # Now unify predictor states by concatenating along batch dim=0
        merged.predictor_primary_key = cls._merge_list_tensors(
            [split.predictor_primary_key for split in splits]
        )
        merged.predictor_primary_value = cls._merge_list_tensors(
            [split.predictor_primary_value for split in splits]
        )
        merged.predictor_importance_key = cls._merge_list_tensors(
            [split.predictor_importance_key for split in splits]
        )

        return merged

    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats)
        self.predictor_primary_key = self._repeat_list_tensors(
            self.predictor_primary_key, repeats
        )
        self.predictor_primary_value = self._repeat_list_tensors(
            self.predictor_primary_value, repeats
        )
        self.predictor_importance_key = self._repeat_list_tensors(
            self.predictor_importance_key, repeats
        )

    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices)
        self.predictor_primary_key = self._select_list_tensors(
            self.predictor_primary_key, indices
        )
        self.predictor_primary_value = self._select_list_tensors(
            self.predictor_primary_value, indices
        )
        self.predictor_importance_key = self._select_list_tensors(
            self.predictor_importance_key, indices
        )

    @staticmethod
    def _ensure_list_capacity(lst: list, idx: int, fill=None):
        if len(lst) <= idx:
            lst.extend([fill] * (idx + 1 - len(lst)))

    @staticmethod
    def _slice_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], start: int, end: int
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t[start:end, ...])
        return out

    @classmethod
    def _merge_list_tensors(
        cls, list_of_lists: List[List[Optional[torch.Tensor]]]
    ) -> List[Optional[torch.Tensor]]:
        # If no splits, return empty
        if not list_of_lists:
            return []

        # Number of layers is length of the sub-list from the first split
        max_len = len(list_of_lists[0])
        merged = [None] * max_len

        for layer_idx in range(max_len):
            # collect that layer_idx from each split
            chunk_tensors = []
            for split in list_of_lists:
                t = split[layer_idx] if layer_idx < len(split) else None
                if t is not None:
                    chunk_tensors.append(t)
            if len(chunk_tensors) == 0:
                merged[layer_idx] = None
            else:
                merged[layer_idx] = torch.cat(chunk_tensors, dim=0)
        return merged

    @staticmethod
    def _repeat_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], repeats: int
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t.repeat_interleave(repeats, dim=0))
        return out

    @staticmethod
    def _select_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], indices: torch.Tensor
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t.index_select(0, indices))
        return out


class TokenImportancePredictorAttentive(nn.Module):
    def __init__(self, config, pred_hid_size, num_heads, num_hidden_layers, dDash, intdim, \
                 attn_reduce_factor, dropout=0.1):
        """
        Optimized Token Importance Predictor with parallel Q-K projections and simplified mapping.
        
        Args:
            config: Configuration object containing model parameters.
            pred_hid_size (int): Hidden size for the predictor's attention layer.
            num_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of transformer layers to predict.
            dropout (float): Dropout probability.
            q_downscale (int): Factor to downscale the Q dimension for efficiency.
            intermediate_dim (int): Intermediate dimension for non-linear transformations in projections.
        """
        super().__init__()
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4) # Predictor head dimension is not the same as the model head dimension.
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False
        assert pred_hid_size % (num_heads * 4) == 0, "pred_hid_size must be divisible by num_heads * 4."

        # Reduce the hidden size for attention computations
        self.hidden_size_reduced = self.hidden_size // self.attn_reduce_factor  # For example, reduce to 1/4th
        assert self.hidden_size_reduced % self.num_heads == 0, "Reduced hidden size must be divisible by num_heads"
        self.attn_head_dim = self.hidden_size_reduced // self.num_heads

        # Input projection to reduce hidden size
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size_reduced, bias=False)

        # Query, Key, Value projections for attention
        self.q_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.k_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.v_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        # Output projection to restore hidden size
        # self.o_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)

        # LayerNorm and Feed-forward network
        self.norm1 = nn.LayerNorm(self.hidden_size_reduced)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.ffn_hidden_size = 2 * self.hidden_size_reduced  # Typical FFN hidden size
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size_reduced, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.hidden_size),
            nn.Dropout(self.dropout)
        )
        # Add extra LayerNorm for the importance branch when not using the old design.
        self.norm_importance = nn.LayerNorm(self.hidden_size)

        # Define Q and K projection layers for all layers in parallel with non-linearity[]
        # Output shape: [B, L, N * H * D']
        self.q_proj_importance = nn.Sequential(
            nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, num_hidden_layers * num_heads * self.dDash, bias=False)
        )
        self.k_proj_importance = nn.Sequential(
            nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.intermediate_dim, num_hidden_layers * num_heads * self.dDash, bias=False)
        )

        # Initialize rotary positional embeddings
        self._init_rope()
        self._initialize_weights()
        self.device = None

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize in_proj_weight
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

                # Initialize out_proj
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def _init_rope(self):

        # send self.config but after modifying head_dim to be self.head_dim just in the function call
        config_copy = copy.deepcopy(self.config)
        config_copy.rope_scaling = {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
        config_copy.head_dim = self.attn_head_dim
        
        # Rotary embedding for attention layer
        self.rotary_emb_attn = LlamaRotaryEmbedding(
            config_copy
        )

        config_copy.head_dim = self.dDash
        # Rotary embedding for importance projection
        self.rotary_emb_importance = LlamaRotaryEmbedding(
            config_copy
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False, layer_idx=None):
        """
        Forward pass for the Optimized Token Importance Predictor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, HQ].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, 1, L] or [B, 1, L, L].
            position_ids (torch.Tensor, optional): Position IDs.
            past_key_value (tuple, optional): Past key and value states.
            use_cache (bool, optional): Whether to use cache.
        
        Returns:
            torch.Tensor: Importance scores of shape [B, N, H, L, L].
        """
        layer_idx = 0 # Guaranteed to be 0, as we only have one predictor!

        # Set device if not already set
        if self.device != hidden_states.device:
            self.device = hidden_states.device
            self.to(self.device)
            
        B, L, E = hidden_states.size()
        
        # Reduce hidden size
        hidden_states = hidden_states.to(self.input_proj.weight.dtype)
        hidden_states_reduced = self.input_proj(hidden_states)  # [B, L, hidden_size_reduced]
        # Compute q, k, v for attention
        q = self.q_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        k = self.k_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        v = self.v_proj_attn(hidden_states_reduced)  # [B, L, hidden_size_reduced]
        # Reshape q, k, v to [B, num_heads, L, attn_head_dim]
        q = q.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        k = k.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        v = v.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)  # [B, num_heads, L, attn_head_dim]
        if (past_key_value is not None
            and layer_idx < len(past_key_value.predictor_primary_key)
            and past_key_value.predictor_primary_key[layer_idx] is not None):
            offset = past_key_value.predictor_primary_key[layer_idx].shape[2]  # old_k.shape[2]
        else:
            offset = 0

        # total seq length for new + old
        kv_seq_len = offset + L

        # Step 2: build position_ids for just the new chunk [offset..offset+L-1]
        if position_ids is None:
            # shape [B, L], e.g. [0..(offset+L-1)]
            position_ids = torch.arange(offset, offset + L, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)

        # Step 3: apply rotary to just the new chunk k,v with the correct offset
        cos, sin = self.rotary_emb_attn(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Step 4: ask the cache to append them.  Then re‐assign k, v to the full cat
        if use_cache and past_key_value is not None:
            k, v = past_key_value.update_predictor_primary(k.detach(), v.detach(), layer_idx)
            kv_seq_len = k.size(2)  # now includes old + new

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_output = attn_output.to(q.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size_reduced)
        attn_output = self.norm1(attn_output)
        ffn_output = self.ffn(attn_output)
        # Temporary measure, till old predictor fully deprecated
        hidden_states = self.norm2(hidden_states + ffn_output)

        B, L, E = hidden_states.size()
        # Importance projections
        H = self.num_heads
        N = self.num_hidden_layers

        hidden_states_for_importance = self.norm_importance(hidden_states)
        q_importance = self.q_proj_importance(hidden_states_for_importance)
        k_importance = self.k_proj_importance(hidden_states_for_importance)

        # Reshape and permute to [B, N, H, L, D']
        q_importance = q_importance.view(B, L, N, H, self.dDash).permute(0, 2, 3, 1, 4).contiguous()  # [B, N, H, L, D']
        k_importance = k_importance.view(B, L, N, H, self.dDash).permute(0, 2, 3, 1, 4).contiguous()  # [B, N, H, L, D']

        # Flatten N and H for efficient computation
        q_importance = q_importance.view(B * N * H, L, self.dDash)  # [BNH, L, D']
        k_importance = k_importance.view(B * N * H, L, self.dDash)  # [BNH, L, D']

        # Apply rotary positional embeddings
        cos, sin = self.rotary_emb_importance(k_importance, position_ids)
        q_importance, k_importance = apply_rotary_pos_emb(q_importance, k_importance, cos, sin, position_ids)

        if use_cache and past_key_value is not None:
            k_importance = past_key_value.update_predictor_importance(k_importance.detach(), layer_idx)
            
        k_importance = k_importance.view(B * H, N, -1, self.dDash)  # [BNH, L, D']
        q_importance = q_importance.view(B * H, N, -1, self.dDash)  # [BH, N, L, D']
        return q_importance, k_importance


def threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len):
    """
    Create a mask tensor based on per-head thresholds, setting values below the threshold to -inf.
    
    Args:
    - unadj_importance_mask: torch.Tensor of shape [B, H, Lq, Lk].
    - perhead_thresholds: torch.Tensor of shape [H], per-head thresholds.
    - min_sparse_index: Minimum index for sparsity; values below this index will not be masked.
    - bsz: Batch size.
    - q_len: Query length (Lq).
    - key_len: Key length (Lk).

    Returns:
    - mask_tensor: torch.Tensor of shape [B, H, Lq, Lk], with values below threshold as -inf.
    """
    # Ensure perhead_thresholds is in the correct shape for broadcasting
    thresholds_broadcast = perhead_thresholds.view(1, -1, 1, 1)  # [1, H, 1, 1]

    # Compare unadj_importance_mask with thresholds to create a mask
    mask_tensor = torch.where(
        unadj_importance_mask >= thresholds_broadcast, 
        torch.zeros_like(unadj_importance_mask), 
        torch.full_like(unadj_importance_mask, float('-inf'))
    )  # [B, H, Lq, Lk]

    # Ensure mask_tensor has mask_tensor[:, :, :, :min_sparse_index] = 0
    mask_tensor[:, :, :, :min_sparse_index] = 0.0

    return mask_tensor

class SlidingWindowCache:
    def __init__(self, max_seq_len, sliding_window, device):
        self.sliding_window = sliding_window
        self.device = device
        if sliding_window is None:
            self.max_seq_len = 0
            self.window = None
        else:
            self.max_seq_len = max_seq_len
            self.window = self._create_window(self.max_seq_len)

    def _create_window(self, seq_len):
        idx = torch.arange(seq_len, device=self.device)
        query = idx.unsqueeze(1)  # [seq_len, 1]
        key = idx.unsqueeze(0)    # [1, seq_len]
        win = (key >= (query - self.sliding_window + 1)) & (key <= query)
        return win.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,seq_len]

    def get_window(self, q_len, key_len):
        if self.sliding_window is None:
            return None
        req = max(q_len, key_len)
        if req > self.max_seq_len:
            self.max_seq_len = req
            self.window = self._create_window(self.max_seq_len)
        return self.window[:, :, :q_len, :key_len]

def enforce_sliding_window(mask_tensor, window):
    if window is None:
        return mask_tensor
    return mask_tensor.masked_fill(window, 0.0)


def sorted_index_to_mask(
    sorted_indices,
    attention_mask,
    min_sparse_index,
    bsz,
    q_len,
    key_len,
    sparse_aggression,
    sliding_window=None
):
    """
    sorted_indices: [B, H, q_len, key_len]
    attention_mask: [1, 1, q_len, key_len]  (True = keep, False = mask out, or vice versa)
    min_sparse_index: guaranteed front region to keep
    sliding_window: guaranteed trailing region (for each query) to keep
    sparse_aggression: float in [0,1], fraction of keys to drop or keep
    """
    device = sorted_indices.device
    dtype = sorted_indices.dtype

    # Step 1: Compute base K
    if q_len == 1:  
        query_positions = torch.arange(q_len, device=device).view(1, 1, q_len, 1).float()
        query_positions[0] = key_len + 1
    else:
        query_positions = torch.arange(q_len, device=device).view(1, 1, q_len, 1).float() + 1.0
    K_original = torch.ceil(query_positions * sparse_aggression).long()  # [1,1,q_len,1]
    K_original = torch.clamp(K_original, max=key_len)

    # Step 1b: Incorporate guaranteed region
    guaranteed = min_sparse_index
    if sliding_window is not None:
        guaranteed += sliding_window
    # Subtract guaranteed from the original K
    K_adjusted = K_original - guaranteed
    # Ensure K_adjusted is at least 0
    K_adjusted = torch.clamp(K_adjusted, min=0, max=key_len)

    # Step 2: Expand attention_mask to [B,H,q_len,key_len]
    attention_mask_expanded = attention_mask.expand(bsz, -1, -1, -1)
    attention_mask_expanded = attention_mask_expanded.expand(-1, sorted_indices.size(1), -1, -1)
    # Convert True -> 1, False -> 0
    attention_mask_expanded = (~attention_mask_expanded.bool()).int()

    # Step 3: Gather (reorder) mask by sorted_indices
    gathered_mask = torch.gather(attention_mask_expanded, dim=-1, index=sorted_indices)

    # Step 4: cumsum along sorted dimension
    gathered_mask_float = gathered_mask.float()
    cum_sum = torch.cumsum(gathered_mask_float, dim=-1)  # [B,H,q_len,key_len]

    # Step 5: Compare cumsum <= K_adjusted
    # Expand K_adjusted to [B,H,q_len,key_len] for broadcast
    K_broadcast = K_adjusted.view(1, 1, q_len, 1).expand_as(cum_sum)
    selected_mask = (cum_sum <= K_broadcast)

    # Step 6: Prepare final mask_tensor with -inf by default
    mask_tensor = torch.full_like(attention_mask_expanded.float(), float('-inf'))

    # Step 7: Scatter 0 where selected, -inf otherwise
    scatter_values = torch.zeros_like(gathered_mask_float)
    scatter_values = scatter_values.masked_fill(~selected_mask, float('-inf'))
    mask_tensor.scatter_(-1, sorted_indices, scatter_values)

    # Step 8: Force the guaranteed front region unmasked
    mask_tensor[:, :, :, :min_sparse_index] = 0.0

    # We do NOT forcibly unmask the trailing `sliding_window` here,
    # because we typically do it with a separate function that
    # ensures the last `sliding_window` positions are unmasked for each query.
    # Replace with self.sliding_window where referenced
    # Where not referenced, reduce budget in calculation.

    return mask_tensor

class FlashInferDynamicCache(PredictorDynamicCache):
    """
    Extended DynamicCache that uses FlashInfer's paged KV cache for faster decoding.
    Designed for batch_size=1 and page_size=1 (one token per page).
    Maintains compatibility with predictor caches.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.decode_wrapper = None
        self.max_seq_len = 8192  # Initial allocation
        self.workspace_buffer = None
        self.indptr_buffer = None
        self.indices_buffer = None 
        self.last_page_len_buffer = None
        self.output = False
        self.kv_cache_layers = []
        self.current_len = 0
        # Track whether to use FlashInfer for specific layers
        self.flashinfer_enabled_layers = set()

    def initialize(self, config, dtype=None, device="cuda:0"):
        """Initialize the flashinfer resources"""
        if self.initialized:
            return
            
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Ensure we're using a supported dtype (float16 or bfloat16)
        if dtype is None or dtype == torch.float32:
            self.dtype = torch.float16  # Default to float16 if not specified or if float32
        else:
            self.dtype = dtype
            
        self.device = device
        self.page_size = 1  # One token per page
        
        # Allocate workspace buffer (128MB)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        
        # Allocate buffers for page table (oversized for growth)
        self.indptr_buffer = torch.zeros(2, dtype=torch.int32, device=device)  # [batch_size + 1]
        self.indices_buffer = torch.arange(self.max_seq_len, dtype=torch.int32, device=device)
        self.last_page_len_buffer = torch.ones(1, dtype=torch.int32, device=device)  # [batch_size]
        
        # Initialize flashinfer wrapper
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "HND"
        )
        
        # Create paged KV cache for each layer
        self.kv_cache_layers = [
            torch.zeros(
                self.max_seq_len, 2, self.num_kv_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=device
            ) for _ in range(self.num_layers)
        ]
        
        self.initialized = True

    def plan(self):
        """Plan the decode operation with current sequence length"""
        if self.current_len == 0:
            return
            
        # Update indptr according to current sequence length
        self.indptr_buffer[0] = 0
        self.indptr_buffer[1] = self.current_len
        
        # Ensure indices are correct (may be unnecessary if using arange)
        if len(self.indices_buffer) < self.current_len:
            self.indices_buffer = torch.arange(self.max_seq_len, dtype=torch.int32, device=self.device)
            
        try:
            token_sparsity = 1.0
            indices_buffer = self.indices_buffer[:int(self.current_len * token_sparsity)]
            # print(f'intptr_buffer: {self.indptr_buffer}')
            # print(f'indices_buffer: {indices_buffer}')
            # print(f'last_page_len_buffer: {self.last_page_len_buffer}')
            # Plan decode with current sequence configuration
            self.decode_wrapper.end_forward()
            self.decode_wrapper.plan(
                self.indptr_buffer,
                indices_buffer,
                self.last_page_len_buffer,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                pos_encoding_mode="ROPE_LLAMA",
                q_data_type=self.dtype,
                kv_data_type=self.dtype
            )
        except Exception as e:
            print(f"Warning: FlashInfer plan failed with error: {e}")
            print(f"Disabling FlashInfer for this batch. Will use standard attention instead.")
            # If planning fails, disable flashinfer for all layers
            self.flashinfer_enabled_layers = set()

    def resize_if_needed(self, target_len):
        """Resize buffers if needed to accommodate longer sequences"""
        if target_len <= self.max_seq_len:
            return
            
        new_max_len = max(self.max_seq_len * 2, target_len)
        
        # Resize indices buffer
        self.indices_buffer = torch.arange(new_max_len, dtype=torch.int32, device=self.device)
        
        # Resize KV cache for each layer
        new_kv_cache_layers = []
        for layer_idx, old_cache in enumerate(self.kv_cache_layers):
            # Create new cache with larger size
            new_cache = torch.zeros(
                new_max_len, 2, self.num_kv_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device
            )
            # Copy old data
            new_cache[:self.current_len] = old_cache[:self.current_len]
            new_kv_cache_layers.append(new_cache)
            
        self.kv_cache_layers = new_kv_cache_layers
        self.max_seq_len = new_max_len

    def update(self, key_states, value_states, layer_idx):
        """
        Update KV cache with new key and value states
        Returns the concatenated KV states
        """
        # input key_states: [1, num_heads, seq_len, head_dim]
        # input value_states: [1, num_heads, seq_len, head_dim]
        # First, use the parent class's standard update logic to maintain HF compatibility
        # and update the basic DynamicCache fields
        if layer_idx >= len(self.key_cache):
            # Extend lists if necessary
            self._ensure_list_capacity(self.key_cache, layer_idx, None)
            self._ensure_list_capacity(self.value_cache, layer_idx, None)
            
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate along the sequence length dimension
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        
        # Now handle FlashInfer-specific updates
        # Ensure cache is initialized
        if not self.initialized:
            config = getattr(self, "config", None)
            if config is None:
                # Extract config from key_states
                batch_size, num_heads, seq_len, head_dim = key_states.shape
                config = type('Config', (), {
                    'num_hidden_layers': len(self.key_cache) if self.key_cache else 32, # default, will be updated later
                    'num_attention_heads': num_heads,
                    'num_key_value_heads': num_heads,  # Assume same as num_heads if not specified
                    'hidden_size': num_heads * head_dim
                })
            # Initialize with the tensor's dtype, but convert float32 to float16 automatically
            dtype = key_states.dtype if key_states.dtype != torch.float32 else torch.float16
            self.initialize(config, dtype=dtype, device=key_states.device)

        # Add this layer to the set of FlashInfer-enabled layers
        self.flashinfer_enabled_layers.add(layer_idx)
        
        # Extract shapes
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Prepare for new tokens
        new_len = self.current_len + seq_len
        self.resize_if_needed(new_len)
        
        # Cast inputs to the correct dtype if needed
        if key_states.dtype != self.dtype:
            key_states = key_states.to(self.dtype)
        if value_states.dtype != self.dtype:
            value_states = value_states.to(self.dtype)
        # Update KV cache for the specific layer
        layer_cache = self.kv_cache_layers[layer_idx]
        # key_states: [1, num_kv_heads, seq_len, head_dim] -> key_reshaped: [seq_len, 2, num_kv_heads, page_size, head_dim]
        # value_states: [1, num_kv_heads, seq_len, head_dim] -> value_reshaped: [seq_len, 2, num_kv_heads, page_size, head_dim]
        key_reshaped = key_states.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        value_reshaped = value_states.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        #layer_cache: [max_seq_len, 2, page_size, num_kv_heads, head_dim]
        # # Update current length for next call
        if layer_idx == 0:  # Only increment once per layer stack
            self.current_len = new_len
            # Update planning for flashinfer
            self.plan()
        if seq_len == 1:
            layer_cache[self.current_len - 1, 0] = key_reshaped[0]
            layer_cache[self.current_len - 1, 1] = value_reshaped[0]
        else:
            start_idx = self.current_len - seq_len
            layer_cache[start_idx:self.current_len, 0] = key_reshaped
            layer_cache[start_idx:self.current_len, 1] = value_reshaped
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_layer_cache(self, layer_idx):
        """Get the flashinfer compatible cache for a specific layer"""
        if not self.initialized or layer_idx >= len(self.kv_cache_layers):
            return None
        return self.kv_cache_layers[layer_idx]
        
    def run_flashinfer_decode(self, q, layer_idx):
        """
        Run flashinfer decode for a specific layer
        q: query tensor [1, num_heads, seq_len, head_dim]
        """
        if not self.initialized or self.current_len == 0 or layer_idx not in self.flashinfer_enabled_layers:
            return None
            
        # Ensure query is in the correct data type
        if q.dtype != self.dtype:
            q = q.to(self.dtype)
        # Get layer's KV cache
        layer_cache = self.kv_cache_layers[layer_idx]
        try:
            
            # Run flashinfer decode
            return self.decode_wrapper.run(
                q.squeeze(-2), 
                layer_cache[:self.current_len]  # Only use valid entries
            )
        except Exception as e:
            print(f"Warning: FlashInfer decode failed with error: {e}")
            print(f"Disabling FlashInfer for layer {layer_idx}. Will use standard attention instead.")
            # If decode fails, disable flashinfer for this layer
            if layer_idx in self.flashinfer_enabled_layers:
                self.flashinfer_enabled_layers.remove(layer_idx)
            return None
        
    def crop(self, max_length: int):
        """Crop cache to max_length"""
        super().crop(max_length)
        
        if self.initialized:
            self.current_len = min(self.current_len, max_length)
            # No need to physically crop the tensors, just update the length


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, config=None):
        self.scaling_factor = scaling_factor
        super().__init__(config)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, config=None):
        self.scaling_factor = scaling_factor
        super().__init__(config)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttentionExperimental(nn.Module):
    def __init__(self, config: LlamaConfig, producer=None, layer_idx=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.inference_mode = True
        self.producer = producer
        self.layer_idx = layer_idx
        self.token_sparse_method = None
        self.sparse_aggression = None
        self.stream_llm_start_size = None
        self.dDash = None
        self.intdim = None
        self.attn_reduce_factor = None
        self.head_attn_reduce_factor = None
        self.effective_sparsity = None
        self.min_sparse_index = None
        self.pred_hid_size = self.hidden_size
        self.num_tok_per_page = None
        self.calc_hitrates = False
        self.flash_attn = False
        self.train_headpredictor = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.old_predictor = None

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.headmseloss = MSELoss(reduction='none')
            self.headmsemagn_loss = None
        
        if self.producer is None:  # This is the producer layer
            self.q_importance = None  # Shared mask across layers during inference
            self.k_importance = None
            self.head_importances = None
            self.actmagn_masklist = {}
            self.available_tokens = {}

        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        
    def update_predictor(self):
        self.sparse_token_predictor = TokenImportancePredictorAttentive(
            self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = self.dDash, \
            intdim = self.intdim, attn_reduce_factor=self.attn_reduce_factor
        ).to('cuda:0')
        self.sparse_token_predictor.flash_attn = self.flash_attn

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        if self.token_sparse_method is not None:
            try:
                mname = self.config._name_or_path.split("/")[-1]
                read_path = f"threshold_calibs/{mname}/{self.token_sparse_method}.pkl"
                threshold_model_dictionary = torch.load(read_path)
                self.tok_calibration_set = threshold_model_dictionary
            except:
                pass
        if self.token_sparse_method == "LazyLLM":
            if self.layer_idx <= 9:
                self.sparse_aggression = 1
            elif self.layer_idx <= 19:
                self.sparse_aggression = 0.7
            elif self.layer_idx <= 28:
                self.sparse_aggression = 0.4
            else:
                self.sparse_aggression = 0.1
        elif "fixed" in self.token_sparse_method:
            if self.layer_idx == 0:
                self.sparse_aggression = 1
            else:
                self.sparse_aggression = 1 - float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
        elif "progressive" in self.token_sparse_method:
            pc_drop = float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
            self.sparse_aggression = (1 - pc_drop) ** (self.layer_idx)  # (x% per layer, progressive_xpc style)
        else:
            raise ValueError(f"Unknown token sparsity method {self.token_sparse_method}")
            

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.config
            )
        else:
            scaling_type = self.config.rope_scaling.get("type") or self.config.rope_scaling.get("rope_type")
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear" or scaling_type == 'llama3':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    config=self.config
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    config=self.config
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[DynamicCache, PredictorDynamicCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PredictorDynamicCache]]:
        bsz, q_len, _ = hidden_states.size()
        Ltrack = hidden_states.size(1)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(q_len * self.sparse_aggression)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # AHMED: Modified this to use the newer version.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        # if self.layer_idx == 0 and q_len != 1:
        #     print(f'key_states: {key_states}')
        # if self.inference_mode and q_len == 1 and self.layer_idx in past_key_value.flashinfer_enabled_layers:
        if self.inference_mode and q_len == 1:
            attn_output = past_key_value.run_flashinfer_decode(query_states, self.layer_idx)
            #[batch_size, num_qo_heads, head_dim] -> [batch_size, q_len, num_qo_heads * head_dim]
            attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim).to(self.o_proj.weight.dtype)
            attn_output = self.o_proj(attn_output)
            attn_weights = None
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            # attn_output = F.scaled_dot_product_attention(
            #     query_states,                
            #     key_states,                  
            #     value_states,                
            #     attn_mask=None,                          
            #     is_causal=False          
            # )
            # attn_output = attn_output.transpose(1, 2).contiguous()
            # attn_output = attn_output.view(bsz, -1, self.hidden_size)
            # attn_output = self.o_proj(attn_output)
            # attn_weights = None
        else:
            kv_seq_len = key_states.shape[-2]
            final_mask = None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_len = key_states.size(2)
            bsz, q_len = query_states.size(0), query_states.size(2)

            if attention_mask is None:
                # We want a [q_len, kv_seq_len] boolean upper-triangular mask
                causal_mask_2d = torch.ones(q_len, kv_seq_len, 
                                            device=hidden_states.device, 
                                            dtype=torch.bool).triu(diagonal=1)
                # Then shape it to [bsz, 1, q_len, kv_seq_len]
                causal_mask_4d = causal_mask_2d.unsqueeze(0).expand(bsz, 1, q_len, kv_seq_len)
                # Now fill -inf where the mask is True
                attention_mask = torch.full_like(causal_mask_4d, 0, dtype=hidden_states.dtype)
                if q_len != 1:
                    attention_mask = attention_mask.masked_fill(causal_mask_4d, float("-inf"))

            if self.inference_mode:
                min_sparse_index = self.min_sparse_index
                with torch.no_grad():
                    if evalmode == "ExpPred":
                        if self.layer_idx > 0:
                            q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                            k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                            importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash) # [BH, Lq, Lk]
                            importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len) # [B, H, Lq, Lk]
                            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                            importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                            sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                            sorted_indices = sorted_indices[:, :, -q_len:, :]
                            mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression, self.sliding_window)
                            ### Threshold variance investigation
                            if self.sliding_window is not None:
                                if not hasattr(self, "window_cache"):
                                    self.window_cache = SlidingWindowCache(max_seq_len=1024,
                                                                        sliding_window=self.sliding_window,
                                                                        device=mask_tensor.device)
                                window = self.window_cache.get_window(q_len, key_len)
                                mask_tensor = enforce_sliding_window(mask_tensor, window)
                            final_mask = mask_tensor

                            self.final_mask_investigate = final_mask
                            attn_weights = attn_weights + mask_tensor + attention_mask
                        else:
                            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                            attn_weights = attn_weights + attention_mask
                    else:
                        raise ValueError(f"Unknown eval mode {evalmode}")
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, -1, self.hidden_size)

                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)
            if self.producer is None:
                try:
                    q_importance, k_importance = self.sparse_token_predictor(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,  # the same single cache
                        use_cache=use_cache,
                        layer_idx=self.layer_idx,       # or pass 0
                    )
                    q_len = attn_output.size(1)
                    k_len = k_importance.size(-1)
                except:
                    print(traceback.format_exc())
                    import pdb; pdb.set_trace()

                self.q_importance = q_importance
                self.k_importance = k_importance


            if not output_attentions:
                attn_weights = None
        return attn_output, attn_weights

def convert_kvcache_experimental(model, config, producer_frequency):
    producer_layer = None
    producer_layer_device = None
    layer_counter = {'idx': 0}

    def recurse_convert(parent_module):
        nonlocal producer_layer
        nonlocal producer_layer_device
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                if layer_counter['idx'] % producer_frequency == 0:
                    new_module = LlamaAttentionExperimental(config).to(dtype).to(device)
                    producer_layer = new_module
                    producer_layer_device = device
                else:
                    new_module = LlamaAttentionExperimental(
                        config,
                        producer=producer_layer,
                        layer_idx=layer_counter['idx']
                    ).to(dtype).to(device)
                new_module.load_state_dict(module.state_dict(), strict=False)
                is_producer = layer_counter['idx'] % producer_frequency == 0
                if is_producer:
                    print(f"Converted Producer layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                else:
                    print(f"Converted layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                parent_module._modules[name] = new_module
                layer_counter['idx'] += 1
    recurse_convert(model)
    producer_layer = producer_layer.to(producer_layer_device)
    return model


# ---------------------------------------------------------------------
# 1) Custom Config subclass
# ---------------------------------------------------------------------
class LlamaButlerConfig(LlamaConfig):
    """
    Extends HF's LlamaConfig to hold optional extra parameters for the "Butler" logic.
    You can store your custom attributes here, so they can be serialized in config.json.
    """

    model_type = "llama_butler"

    def __init__(
        self,
        eval_llm_mode="ExpPred",
        token_sparse_method="fixed_50pc",
        producer_frequency=8,
        dDash=16,
        attn_reduce_factor=4,
        head_attn_reduce_factor=4,
        intdim=256,
        flash_attn=False,
        train_headpredictor=False,
        min_sparse_index=5,
        lookahead=0,
        sliding_window=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eval_llm_mode = eval_llm_mode
        self.token_sparse_method = token_sparse_method
        self.producer_frequency = producer_frequency
        self.dDash = dDash
        self.attn_reduce_factor = attn_reduce_factor
        self.head_attn_reduce_factor = head_attn_reduce_factor
        self.intdim = intdim
        self.flash_attn = flash_attn
        self.train_headpredictor = train_headpredictor
        self.min_sparse_index = min_sparse_index
        self.lookahead = lookahead
        self.sliding_window = sliding_window


# ---------------------------------------------------------------------
# 2) The main Butler model class
# ---------------------------------------------------------------------
class LlamaButlerForCausalLM(LlamaForCausalLM):
    """
    A subclass of HF's LlamaForCausalLM that:
      - Patches each LlamaAttention to your LlamaAttentionExperimental
      - Sets specialized attributes (eval_llm_mode, etc.)
      - Overrides _prepare_cache_for_generation to inject PredictorDynamicCache
    """

    # Let HF auto-detect this config class from config.json:
    config_class = LlamaButlerConfig

    def __init__(self, config: LlamaButlerConfig):
        super().__init__(config)
        """
        HF's LlamaForCausalLM initializes:
          self.model = LlamaModel(config)
          self.lm_head = nn.Linear(...)
        """

        # 1) Patch the underlying LlamaModel to replace LlamaAttention with LlamaAttentionExperimental
        self.model = convert_kvcache_experimental(
            self.model,
            config,
            config.producer_frequency
        )

        # 2) Optionally, set per-module attributes so each LlamaAttentionExperimental knows about them:
        for module in self.model.modules():
            if module.__class__.__name__.endswith("AttentionExperimental"):
                # Set these from your config. Or you can hardcode them if you prefer.
                module.eval_llm_mode = config.eval_llm_mode
                module.token_sparse_method = config.token_sparse_method
                module.set_token_sparsity()  # e.g. sets module.sparse_aggression

                module.producer_frequency = config.producer_frequency
                module.dDash = config.dDash
                module.attn_reduce_factor = config.attn_reduce_factor
                module.head_attn_reduce_factor = config.head_attn_reduce_factor
                module.intdim = config.intdim
                module.flash_attn = config.flash_attn
                module.train_headpredictor = config.train_headpredictor
                module.min_sparse_index = config.min_sparse_index
                module.lookahead = config.lookahead
                module.sliding_window = config.sliding_window
                module.num_layers_pred = config.producer_frequency  # example usage

                # If this is a "producer layer" (mod.layer_idx % freq == 0), run update_predictor():
                if hasattr(module, "layer_idx") and (module.layer_idx % config.producer_frequency == 0):
                    module.update_predictor()

        # 3) Patch the dynamic cache (past_key_values) creation. For your evaluation modes:
        if config.eval_llm_mode in ["ExpPred", "ReplAttn"]:
            self._prepare_cache_for_generation = self._patched_prepare_cache_for_generation.__get__(
                self, self.__class__
            )

    # -----------------------------------------------------------------
    # 3) The custom `_prepare_cache_for_generation` override
    # -----------------------------------------------------------------
    def _patched_prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        *args,
        **kwargs
    ):
        """
        This override injects a PredictorDynamicCache
        in place of the standard 'past_key_values'.
        """
        if "past_key_values" not in model_kwargs or model_kwargs["past_key_values"] is None:
            cache = FlashInferDynamicCache()
            cache.config = self.config
            model_kwargs["past_key_values"] = cache
        return model_kwargs