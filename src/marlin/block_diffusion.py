"""Block-diffusion primitives: block ids, attention masks, masked-diffusion noising and loss.

The masks are additive ``[B, 1, Lq, Lk]`` tensors that the BERT backbone adds to its attention scores.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

NEG = -1e9  # added to disallowed attention scores


def make_block_ids(seq_len: int, block_size: int, device=None) -> torch.Tensor:
    """Block index of every position, ``[L]``."""
    return torch.arange(seq_len, device=device) // int(block_size)


def build_block_causal_mask(
    block_id: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    batch_size: int,
    device,
) -> torch.Tensor:
    """Block-causal mask ``[B, 1, L, L]``: attend within the block and to earlier blocks, never to padding."""
    L = block_id.numel()
    allow = block_id.unsqueeze(0) <= block_id.unsqueeze(1)
    allow = allow.unsqueeze(0).expand(batch_size, L, L).clone()
    if pad_mask is not None:
        key_ok = pad_mask.to(torch.bool).unsqueeze(1)
        allow = allow & key_ok
    add = torch.where(allow, torch.zeros((), device=device), torch.full((), NEG, device=device))
    return add.unsqueeze(1)


def build_two_stream_mask(
    block_id: torch.Tensor,
    pad_mask: Optional[torch.Tensor],
    batch_size: int,
    device,
) -> torch.Tensor:
    """Two-stream training mask ``[B, 1, 2L, 2L]`` over ``[clean x0 | noised xt]``.

    ``x0`` attends to ``x0`` in its own and earlier blocks; ``xt`` attends to ``x0`` in earlier
    blocks and to ``xt`` in its own block.
    """
    L = block_id.numel()
    bid_q = block_id.unsqueeze(1)
    bid_k = block_id.unsqueeze(0)

    x0_to_x0 = bid_k <= bid_q
    xt_to_x0 = bid_k < bid_q
    xt_to_xt = bid_k == bid_q
    false_blk = torch.zeros((L, L), dtype=torch.bool, device=device)

    top = torch.cat([x0_to_x0, false_blk], dim=1)      # x0 queries
    bottom = torch.cat([xt_to_x0, xt_to_xt], dim=1)    # xt queries
    allow = torch.cat([top, bottom], dim=0)
    allow = allow.unsqueeze(0).expand(batch_size, 2 * L, 2 * L).clone()

    if pad_mask is not None:
        key_ok = torch.cat([pad_mask.to(torch.bool), pad_mask.to(torch.bool)], dim=1)
        allow = allow & key_ok.unsqueeze(1)

    add = torch.where(allow, torch.zeros((), device=device), torch.full((), NEG, device=device))
    return add.unsqueeze(1)


def sample_block_times(
    batch_size: int, num_blocks: int, device, eps: float = 1e-3, antithetic: bool = True
) -> torch.Tensor:
    """Diffusion time per example and block in ``(eps, 1]``, ``[B, K]``."""
    if antithetic and batch_size % 2 == 0:
        half = torch.rand(batch_size // 2, num_blocks, device=device)
        t = torch.cat([half, 1.0 - half], dim=0)
    else:
        t = torch.rand(batch_size, num_blocks, device=device)
    return t.clamp(eps, 1.0)


def times_to_positions(t_per_block: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
    """Expand block times ``[B, K]`` to positions ``[B, L]``."""
    return t_per_block[:, block_id]


def noise_masked(
    input_ids: torch.Tensor,
    t_pos: torch.Tensor,
    mask_index: int,
    noiseable: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask each noiseable position with probability equal to its block's time; returns ``(xt, mask)``."""
    u = torch.rand_like(t_pos)
    mask_bool = (u < t_pos) & noiseable.to(torch.bool)
    xt = torch.where(mask_bool, torch.full_like(input_ids, int(mask_index)), input_ids)
    return xt, mask_bool


def diffusion_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask_bool: torch.Tensor,
    t_pos: torch.Tensor,
    block_id: torch.Tensor,
    num_blocks: int,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``1/t``-weighted cross-entropy over masked positions, and the unweighted loss per block (detached)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    m = mask_bool.float()
    w = 1.0 / t_pos.clamp_min(eps)
    weighted = nll * w * m
    denom = m.sum().clamp_min(1.0)
    loss = weighted.sum() / denom

    per_block = []
    for k in range(num_blocks):
        blk = (block_id == k).float().unsqueeze(0)
        bm = m * blk
        d = bm.sum().clamp_min(1.0)
        per_block.append((nll * bm).sum() / d)
    return loss, torch.stack(per_block).detach()
