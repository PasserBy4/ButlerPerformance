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



class HeadImportancePredictor(nn.Module):
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
        self.is_head_predictor = None
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4)
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False

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

        self.ffn_hidden_size = 4 * self.hidden_size_reduced  # Typical FFN hidden size
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size_reduced, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.num_heads * self.num_hidden_layers),
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
        config_copy = copy.deepcopy(self.config)
        config_copy.head_dim = self.attn_head_dim
        # Rotary embedding for attention layer
        self.rotary_emb_attn = LlamaRotaryEmbedding(
            config_copy
        )
        # Rotary embedding for importance projection
        self.rotary_emb_importance = LlamaRotaryEmbedding(
            config_copy
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
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
        # Set device if not already set
        if self.device != hidden_states.device:
            self.device = hidden_states.device
            self.to(self.device)

        B, L, E = hidden_states.size()
        if past_key_value is None:
            past_key_value = {}
        # if L == 1:
        #     import pdb; pdb.set_trace()
        past_primary = past_key_value.get('primary', None)
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
        # Compute kv_seq_len before concatenation
        if past_primary is not None:
            past_L = past_primary[0].shape[2]
            kv_seq_len = past_L + L
        else:
            kv_seq_len = L
        
        # Apply rotary positional embeddings based on kv_seq_len
        cos, sin = self.rotary_emb_attn(v, position_ids)
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(B, kv_seq_len)
        
        if past_primary is not None:
            # Concatenate past k and v
            k = torch.cat([past_primary[0], k], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
            v = torch.cat([past_primary[1], v], dim=2)  # [B, num_heads, past_L + L, attn_head_dim]
        
        # Apply rotary embeddings after concatenation
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Update cache if use_cache is True
        if use_cache:
            past_key_value['primary'] = (k.detach(), v.detach())

        # if self.flash_attn:
        #     sm_scale = 1.0 / math.sqrt(self.attn_head_dim)
        #     attn_output = attention(q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16), True, sm_scale).to(q.dtype)
        # else:
        #     attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_output = attn_output.to(q.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size_reduced)
        attn_output = self.norm1(attn_output)
        head_importances = self.ffn(attn_output)
        return head_importances, past_key_value

def calculate_hit_metrics(estimated_importance: torch.Tensor, 
                          true_importance: torch.Tensor, 
                          top_k_ratio: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate hit accuracy, mean, and max rank correlation between estimated and true importance tensors.
    We compute metrics along the last dimension of the input tensors.

    Shapes:
      - 4D token-importance: [B, H, L, L]. We slice the last query (index -1) => [B, H, L].
      - 3D head-importance:  [B, L, H]. We use all of it as-is => [B, L, H].
    
    Args:
        estimated_importance (torch.Tensor): [B, H, L, L] or [B, L, H]
        true_importance      (torch.Tensor): [B, H, L, L] or [B, L, H]
        top_k_ratio (float): Fraction of top-k elements to consider for hit accuracy (default=0.5).
    
    Returns:
        (hit_accuracy, mean_corr, max_corr):
            hit_accuracy (float): Intersection ratio of top-k sets (0..1).
            mean_corr (float): Average Spearman rank correlation over all [B, ...].
            max_corr (float): Maximum Spearman rank correlation among all [B, ...].
    """

    # 1) Standardize shapes so the last dimension is what we rank over.
    if estimated_importance.dim() == 4:
        # Shape is [B, H, L, L] => slice to keep only the last query => [B, H, L]
        estimated_importance = estimated_importance[:, :, -1, :]
        true_importance      = true_importance[:, :, -1, :]
        # after slicing: [B, H, L]
        # For intersection denominator => top_k * B * H
        denom_for_hits = estimated_importance.size(0) * estimated_importance.size(1)
    elif estimated_importance.dim() == 3:
        # Shape is [B, L, H], the last dimension is H
        # For intersection denominator => top_k * B * L
        denom_for_hits = estimated_importance.size(0) * estimated_importance.size(1)
    else:
        raise ValueError("Tensors must be either 4D [B,H,L,L] or 3D [B,L,H].")

    # 2) Compute Spearman rank correlation along the last dimension.
    #    Sort indices in descending order => get 'ranks' for correlation.
    _, sorted_esti = torch.sort(estimated_importance, dim=-1, descending=True)
    _, sorted_true = torch.sort(true_importance, dim=-1, descending=True)

    # Spearman's rho = 1 - 6 * sum(d^2) / [n*(n^2 - 1)]
    n = sorted_esti.shape[-1]
    d = sorted_esti.float() - sorted_true.float()
    d_squared = d ** 2
    sum_d_squared = d_squared.sum(dim=-1)
    rank_corr = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))  # shape: [B,H] or [B,L]

    mean_corr = rank_corr.mean().item()
    max_corr  = rank_corr.max().item()

    # 3) Compute top-k hit accuracy along the last dimension.
    top_k = max(1, int(n * top_k_ratio))
    _, top_esti_indices = torch.topk(estimated_importance, top_k, dim=-1)
    _, top_true_indices = torch.topk(true_importance,      top_k, dim=-1)

    # top_esti_indices => [B,H,top_k] or [B,L,top_k]
    # top_true_indices => [B,H,top_k] or [B,L,top_k]
    # matches => [B,H,top_k,top_k] or [B,L,top_k,top_k]
    matches = (top_esti_indices.unsqueeze(-1) == top_true_indices.unsqueeze(-2))
    intersection = matches.any(dim=-1).sum(dim=-1)  # => [B,H] or [B,L]

    # Each [B,H] or [B,L] element can have at most 'top_k' matches, so total is top_k * denom_for_hits.
    total_possible = top_k * denom_for_hits
    hit_accuracy = intersection.sum().item() / total_possible  # => 0..1

    return hit_accuracy, mean_corr, max_corr


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
        self.inference_mode = False
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
        if self.train_headpredictor:
            self.sparse_head_predictor = HeadImportancePredictor(
                self.config, self.pred_hid_size, self.num_heads, self.num_layers_pred, dropout=0.1, dDash = self.dDash, \
                intdim = self.intdim, attn_reduce_factor=self.head_attn_reduce_factor
            ).to('cuda:0')
            self.sparse_head_predictor.flash_attn = self.flash_attn

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
                        if self.calc_hitrates:
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=importance_mask,
                                true_importance=attn_weights,
                                top_k_ratio=0.5
                            )
                        if self.calibrate_thresholds:
                            ### Threshold variance investigation
                            unadj_importance_mask = importance_mask.clone()
                            importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                            sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                            sorted_indices = sorted_indices[:, :, -q_len:, :]
                            sorted_values, sorted_ix = torch.sort(importance_mask, dim=-1)
                            sorted_true_values, _ = torch.sort(torch.gather(unadj_importance_mask, dim=-1, index=sorted_ix), dim=-1)
                            true_thresholds = sorted_true_values[:, :, :, int(importance_mask.size(-1) * self.sparse_aggression)]
                            thresholds = sorted_values[:, :, :, int(importance_mask.size(-1) * self.sparse_aggression)]
                            self.true_threshmean = true_thresholds
                            self.threshmean = thresholds
                        if self.test_with_thresholds:
                            unadj_importance_mask = importance_mask.clone()
                            perhead_thresholds = self.tok_calibration_set[self.layer_idx - 1].to(unadj_importance_mask.device) # 0 does not have calibration data.
                            mask_tensor = threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len)
                        else:
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

        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)   
            if self.layer_idx > 0:
                q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash) # [BH, Lq, Lk]
                importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len) # [B, H, Lq, Lk]

                if self.lookahead == 0:
                    self.msemagn_loss = self.mseloss(attn_weights, importance_mask)
                else:
                    self.msemagn_loss = self.mseloss(attn_weights[:, :, self.lookahead:, :], importance_mask[:, :, :-self.lookahead, :])
                self.msemagn_loss = (self.msemagn_loss).mean(dim=(-1, -2))
                self.msemagn_loss = self.msemagn_loss.mean()

                if self.calc_hitrates:
                    self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                        estimated_importance=importance_mask,
                        true_importance=attn_weights,
                        top_k_ratio=0.5
                    )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        if self.layer_idx > 0 and self.train_headpredictor:
            head_importance_tensor = self.producer.head_importances[:, :, :, self.layer_idx % self.producer_frequency].float().to(attn_output.device)
            attn_head_weights = attn_output.mean(dim=-1).permute(0, 2, 1)
            self.headmsemagn_loss = self.headmseloss(attn_head_weights, head_importance_tensor).mean()

            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = calculate_hit_metrics(
                    estimated_importance=head_importance_tensor,
                    true_importance=attn_head_weights,
                    top_k_ratio=0.5
                )
        else:
            self.headmsemagn_loss = 0
            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = 0, 0, 0

            
        checkeverytime = hasattr(self, 'test_with_thresholds')
        if checkeverytime:
            checkeverytime = self.test_with_thresholds
        if final_mask is not None:
            if self.effective_sparsity is None or checkeverytime:
                true_mask = final_mask + attention_mask
                num_deact = true_mask.bool().sum(dim=-1)                   # Number of tokens disabled.
                causally_deact = (attention_mask.bool()).sum(dim=-1).expand_as(num_deact)        # Number of tokens disabled causally anyway
                additional_deact = (num_deact - causally_deact)
                num_active = (~attention_mask.bool()).sum(dim=-1).expand_as(num_deact)    # Number of tokens active at this position if zero-sparsity
                effective_sparsity = 100 * (additional_deact.float() / num_active.float()).mean().item()
                self.effective_sparsity = effective_sparsity
                print("Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)
        if self.layer_idx == 0:
            if self.effective_sparsity is None:
                self.effective_sparsity = 0.0

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
                if self.train_headpredictor:
                    head_importances, past_key_value_hp = self.sparse_head_predictor(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value_hp,
                        use_cache=use_cache
                    )
                    head_importances = head_importances.view(bsz, q_len, self.num_heads, self.num_hidden_layers) # [B L H N]
                q_len = attn_output.size(1)
                k_len = k_importance.size(-1)
            except:
                print(traceback.format_exc())
                import pdb; pdb.set_trace()

            self.q_importance = q_importance
            self.k_importance = k_importance

            if self.train_headpredictor:
                if self.head_importances is None:
                    self.head_importances = head_importances
                else:
                    self.head_importances = torch.cat([self.head_importances, head_importances], dim=1)
        
        # if self.layer_idx == 31:
        #     if q_len == 1:
        #         self.dtok += 1
        #         print(f"Primary Key-Value Shape: {past_key_value.predictor_primary_key[0].shape}, Importance: {past_key_value.predictor_importance_key[0].shape}, Tok-Decoded: {self.dtok}")
        #     else:
        #         self.dtok = 0

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
            model_kwargs["past_key_values"] = PredictorDynamicCache()
        return model_kwargs