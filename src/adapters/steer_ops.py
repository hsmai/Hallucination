"""Omni-Steering 프리미티브 — 선배 probe 코드 이식 (원본 무수정, 로직 복사).

이식 원본 (/home3/t202401082/omni-steering, 2026-07-18 판독):
- ablation.make_rebalance_hook / _locate_attention_mask / _build_causal_mask
  : additive attention mask에 (q_idx→k_idx) 블록을 -inf로 — "attention key masking"
- leak_causal._make_pos_hook : layer pre-hook로 residual의 특정 위치에 벡터 덧셈
- cmm_contam._fwd 구조     : 지정 layer pre-hook로 answer 위치 hidden 캡처

전부 모델-무관 훅이라 양 어댑터(Qwen thinker / VideoLLaMA2 backbone)가 공유한다.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def locate_attention_mask(args, kwargs):
    """(where, key_or_index, mask) — 4D additive attention mask 위치 탐색 (이식)."""
    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        return ("kw", "attention_mask", kwargs["attention_mask"])
    if len(args) >= 2 and torch.is_tensor(args[1]):
        return ("pos", 1, args[1])
    return (None, None, None)


def build_causal_mask(hidden_states, dtype):
    """[1,1,N,N] additive causal mask — SDPA가 mask=None(is_causal)로 도는 경우 폴백 (이식)."""
    n = hidden_states.shape[1]
    neg = torch.finfo(dtype).min
    m = torch.full((n, n), neg, device=hidden_states.device, dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, n, n)


def make_mask_hook(q_idx: Sequence[int], k_idx: Sequence[int], scale: float = 0.0):
    """self_attn forward_pre_hook: (q→k) 블록을 additive mask로 억제/제거 (이식: make_rebalance_hook).

    scale=0 → -inf(완전 제거), 0<scale<1 → log(scale) 가산(억제)."""
    a = torch.as_tensor(list(q_idx), dtype=torch.long)
    v = torch.as_tensor(list(k_idx), dtype=torch.long)
    add = float("-inf") if scale <= 0 else math.log(scale)

    def pre_hook(module, args, kwargs):
        where, key, mask = locate_attention_mask(args, kwargs)
        if mask is None:
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None:
                return args, kwargs
            mask = build_causal_mask(hs, next(module.parameters()).dtype)
            where, key = "kw", "attention_mask"
        m = mask.clone()
        aa, vv = a.to(m.device), v.to(m.device)
        if add == float("-inf"):
            m[:, :, aa.unsqueeze(-1), vv] = torch.finfo(m.dtype).min
        else:
            m[:, :, aa.unsqueeze(-1), vv] = m[:, :, aa.unsqueeze(-1), vv] + add
        if where == "kw":
            kwargs[key] = m
        else:
            args = list(args)
            args[key] = m
            args = tuple(args)
        return args, kwargs

    return pre_hook


def make_inject_hook(vec: torch.Tensor, pos_idx: Sequence[int]):
    """decoder layer forward_pre_hook: residual의 pos_idx 위치에 vec 덧셈 (이식: _make_pos_hook)."""
    p = torch.as_tensor(list(pos_idx), dtype=torch.long)

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None:
            return args, kwargs
        hs = hs.clone()
        pp = p.to(hs.device)
        hs[:, pp, :] = hs[:, pp, :] + vec.to(hs.dtype).to(hs.device)
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs
        else:
            args = (hs,) + tuple(args[1:])
        return args, kwargs

    return pre_hook


def make_capture_hook(store: dict, pos: int):
    """decoder layer forward_pre_hook: 해당 layer 입력 hidden의 pos 위치를 store['h']에 캡처 (이식: _fwd._pre)."""

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None:
            store["h"] = hs[0, pos].detach().float().cpu()
        return None

    return pre_hook
