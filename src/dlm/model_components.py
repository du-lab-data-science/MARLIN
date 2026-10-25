# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import math


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for conditioning SAFE tokens on formula sequence.
    
    This layer allows SAFE token representations (queries) to attend to
    formula atom embeddings (keys/values).
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1):
        """
        Initialize CrossAttentionLayer.
        
        Args:
            hidden_size: Dimension of hidden representations
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query projection (from SAFE tokens)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        
        # Key and Value projections (from formula sequence)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_size)
    
    def forward(self,
                hidden_states: torch.Tensor,
                condition_embeddings: torch.Tensor,
                condition_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply cross-attention.
        
        Args:
            hidden_states: SAFE token hidden states (batch, safe_len, hidden_size)
            condition_embeddings: Conditioning sequence embeddings (batch, cond_len, hidden_size)
            condition_mask: Attention mask for conditioning tokens (batch, cond_len)
        
        Returns:
            Output hidden states (batch, safe_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        
        # Project queries from SAFE tokens
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # Project keys and values from formula
        key_layer = self.transpose_for_scores(self.key(condition_embeddings))
        value_layer = self.transpose_for_scores(self.value(condition_embeddings))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply conditioning mask if provided
        if condition_mask is not None:
            # Reshape mask for broadcasting: (batch, 1, 1, cond_len)
            extended_mask = condition_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask to attention mask (1 -> 0, 0 -> -10000)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask
        
        # Normalize attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Output projection
        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        
        # Residual connection and layer norm
        output = self.output_layer_norm(hidden_states + output)
        
        return output


class SetSelfAttention(nn.Module):
    """
    Permutation-equivariant self-attention for set-structured data.
    No position embeddings - treats input as an unordered set.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            Output hidden states (batch, seq_len, hidden_size)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            # Create attention mask: (batch, 1, 1, seq_len) for keys
            extended_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        output = self.output_dense(context_layer)
        output = self.output_dropout(output)
        output = self.output_layer_norm(hidden_states + output)

        return output


class FingerprintSequenceEncoder(nn.Module):
    """
    Set-aware encoder for Morgan fingerprints.

    Treats fingerprint as an unordered set of active bits:
    - Each bit index has a learnable embedding (semantic meaning of substructure)
    - NO position embeddings (set has no order)
    - Self-attention learns combinatorial relationships between substructures
    """

    def __init__(self,
                 num_bits: int = 4096,
                 embedding_dim: int = 768,
                 max_seq_len: int = 256,
                 activation_threshold: float = 0.0,
                 dropout: float = 0.1,
                 num_self_attention_layers: int = 2,
                 num_attention_heads: int = 12,
                 use_value_projection: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.activation_threshold = activation_threshold
        self.num_self_attention_layers = num_self_attention_layers

        # Bit embeddings: each bit index maps to a unique embedding
        # This encodes "which substructure" semantically
        self.bit_embeddings = nn.Embedding(num_bits, embedding_dim)

        # Layer norm applied after embedding lookup
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Self-attention layers for learning combinatorial relationships
        self.self_attention_layers = nn.ModuleList([
            SetSelfAttention(
                hidden_size=embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout
            )
            for _ in range(num_self_attention_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.bit_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, fingerprint: torch.Tensor) -> tuple:
        """
        Encode fingerprint as a set of active bit embeddings.

        Args:
            fingerprint: Tensor of shape (batch_size, num_bits) - binary Morgan FP

        Returns:
            embeddings: (batch_size, seq_len, embedding_dim) - set embeddings
            mask: (batch_size, seq_len) - 1 for valid bits, 0 for padding
        """
        if fingerprint is None:
            raise ValueError("Fingerprint tensor is required for sequence encoding.")

        if not torch.is_tensor(fingerprint):
            fingerprint = torch.as_tensor(fingerprint, dtype=torch.float32)

        fingerprint = fingerprint.to(dtype=torch.float32)

        single_example = fingerprint.dim() == 1
        if single_example:
            fingerprint = fingerprint.unsqueeze(0)

        batch_size, num_bits = fingerprint.shape
        device = fingerprint.device

        # Find active bits (value > threshold) for each sample
        active_mask = fingerprint > self.activation_threshold  # (batch, num_bits)
        
        # Count active bits per sample
        num_active_per_sample = active_mask.sum(dim=1)  # (batch,)
        max_active = min(self.max_seq_len, int(num_active_per_sample.max().item()))
        max_active = max(max_active, 1)  # At least 1 to avoid empty sequences

        # Vectorized approach: use topk on the mask to get indices
        # For binary mask, topk will give us the indices of 1s
        # We add small noise to break ties consistently
        scores = active_mask.float()
        if self.training:
            # During training, add noise for random sampling when truncating
            scores = scores + torch.rand_like(scores) * 0.5
        
        # Get top-k indices (these are the active bit indices we want)
        _, topk_indices = torch.topk(scores, k=max_active, dim=1, sorted=False)
        
        # Create mask based on actual number of active bits
        # topk_indices contains indices, but some might be from padding (0 values)
        # We need to mask based on original active counts
        position_indices = torch.arange(max_active, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (position_indices < num_active_per_sample.unsqueeze(1)).float()
        
        # For samples with no active bits, ensure at least one position for stability
        # but keep it masked
        no_active = num_active_per_sample == 0
        if no_active.any():
            topk_indices[no_active, 0] = 0  # Use bit 0 as dummy
            mask[no_active] = 0.0

        # Lookup bit embeddings using the actual bit indices
        embeddings = self.bit_embeddings(topk_indices)  # (batch, max_active, embedding_dim)

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Apply self-attention layers to learn combinatorial relationships
        for self_attn in self.self_attention_layers:
            embeddings = self_attn(embeddings, mask)

        # Zero out padding positions
        embeddings = embeddings * mask.unsqueeze(-1)

        if single_example:
            embeddings = embeddings.squeeze(0)
            mask = mask.squeeze(0)

        return embeddings, mask
