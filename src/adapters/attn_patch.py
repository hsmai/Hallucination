"""Qwen2.5-Omni thinker용 eager attention 패치 — AVCD Qwen 포팅의 핵심 (서버 전용).

원리: transformers ≥4.50의 Qwen2.5-Omni modeling 파일은 attn_implementation="eager"일 때
모듈 수준 함수 `eager_attention_forward(module, q, k, v, mask, scaling, ...)`를 호출한다.
이 함수를 런타임에 래핑하여 (외부 repo 무수정):
  1) 기록 모드: 각 thinker text layer의 마지막 query attention 행 (H, S)를 수집
     → src/adapters/common.py 의 dominance_and_threshold 입력
  2) 마스킹 모드: mask_spec/threshold/spans 가 설정돼 있으면 prefill에서
     common.mask_attention_rows 를 적용 (마지막 layer 제외)

⚠ SERVER-UNVERIFIED: transformers 버전에 따라 함수 위치/시그니처가 다를 수 있다.
   S1 스모크에서 patch_qwen25_omni()의 assert가 먼저 실패하도록 방어적으로 작성.
   (runbook 체크리스트: transformers.__version__, 패치 대상 함수 존재, layer 수)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .common import mask_attention_rows

logger = logging.getLogger(__name__)


@dataclass
class AVCDPatchContext:
    """패치된 attention이 참조하는 전역 상태 (프로세스당 1개)."""
    enabled: bool = False                 # False면 원본 동작 그대로
    recording: bool = False               # True면 last-query 행 수집
    span_mask: Optional[torch.Tensor] = None   # (S,) bool — 마스킹 대상 (None=마스킹 없음)
    threshold: float = 0.0
    num_layers: int = 0                   # thinker text layer 수 (마지막 layer 제외용)
    layer_module_ids: frozenset = frozenset()  # thinker text self_attn 모듈 id 집합
    recorded: Dict[int, torch.Tensor] = field(default_factory=dict)  # layer_idx -> (H, S)
    # 구식(4.52.x) 경로 head-chunk 설정: 긴 시퀀스에서 [H,N,N] 피크 제한 (24GB 대응, 수치 동일)
    # 2026-07-17 게이트: chunk=4로도 CMM 최장 클립 OOM(19/200) → 1로 하향 (피크 ~1.2GB)
    chunk_seq_threshold: int = 4096
    head_chunk: int = 1

    def reset_records(self):
        self.recorded = {}

    def records_in_order(self) -> List[torch.Tensor]:
        return [self.recorded[i] for i in sorted(self.recorded)]


CTX = AVCDPatchContext()


def _wrapped_eager(orig_fn):
    def eager_with_avcd(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        # thinker text layer가 아니거나 비활성화면 원본 그대로
        if not CTX.enabled or id(module) not in CTX.layer_module_ids:
            return orig_fn(module, query, key, value, attention_mask,
                           scaling=scaling, dropout=dropout, **kwargs)

        # ---- 원본 eager 수식 재현 (transformers eager_attention_forward 표준형) ----
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv  # GQA 확장
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_weights = _avcd_inject(module, attn_weights, attn_weights.shape[-2])

        attn_weights = attn_weights.to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    eager_with_avcd.__avcd_patched__ = True
    return eager_with_avcd


def install_patch(modeling_module, layer_self_attns) -> AVCDPatchContext:
    """임의의 transformers modeling 모듈에 패치 설치 (테스트 가능한 공통 진입점).

    modeling_module: eager_attention_forward를 가진 모듈 객체
    layer_self_attns: 패치 대상 decoder layer들의 self_attn 모듈 리스트 (순서 = layer_idx 순)
    """
    import transformers

    assert hasattr(modeling_module, "eager_attention_forward"), (
        f"transformers {transformers.__version__}: {modeling_module.__name__}에 "
        "eager_attention_forward가 없습니다 — 패치 지점 변경 필요 (runbook '패치 검증' 항목)")

    CTX.layer_module_ids = frozenset(id(m) for m in layer_self_attns)
    CTX.num_layers = len(layer_self_attns)

    if not getattr(modeling_module.eager_attention_forward, "__avcd_patched__", False):
        modeling_module.eager_attention_forward = _wrapped_eager(
            modeling_module.eager_attention_forward)
        logger.info("%s eager attention 패치 완료 (%d layers)",
                    modeling_module.__name__, CTX.num_layers)
    return CTX


def _avcd_inject(self_attn, attn_weights: torch.Tensor, q_len: int) -> torch.Tensor:
    """softmax 직후 삽입점 — 기록/마스킹 공통 로직 (신식·구식 경로가 공유)."""
    layer_idx = getattr(self_attn, "layer_idx", None)
    if q_len <= 1 or layer_idx is None:
        return attn_weights
    if CTX.recording:
        CTX.recorded[layer_idx] = attn_weights[0, :, -1, :].detach().to(torch.float32).cpu()
    if CTX.span_mask is not None and layer_idx < CTX.num_layers - 1:  # 마지막 layer 제외
        S = attn_weights.shape[-1]
        if CTX.span_mask.shape[0] != S:
            raise RuntimeError(
                f"AVCD span_mask 길이({CTX.span_mask.shape[0]}) != attention S({S}) — "
                f"span 계산이 실제 시퀀스와 어긋남 (runbook T3)")
        dtype = attn_weights.dtype
        # 메모리: fp32 복사 1개만 유지 — 원본은 즉시 해제, 마스킹은 in-place (수식 동일)
        aw_f32 = attn_weights[0].to(torch.float32)
        span = CTX.span_mask.to(attn_weights.device)
        del attn_weights
        attn_weights = mask_attention_rows(
            aw_f32, span, CTX.threshold, inplace=True,
        ).unsqueeze(0).to(dtype)
    return attn_weights


def _legacy_forward_factory(m):
    """transformers 4.52.x 구식 Qwen2_5OmniAttention.forward 교체본.

    본체는 서버 설치본(4.52.3) 소스의 **verbatim 복사**이며, softmax 직후
    _avcd_inject() 한 줄만 삽입한다 (CTX 비활성 시 원본과 동일 동작).
    """
    import math
    from torch import nn

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = m.apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = m.repeat_kv(key_states, self.num_key_value_groups)
        value_states = m.repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] \
            if attention_mask is not None else None
        scale = 1.0 / math.sqrt(self.head_dim)
        n_heads = query_states.shape[1]
        S = key_states.shape[-2]

        # head-chunk: 긴 시퀀스에서 [H,N,N] fp32 피크를 제한 (수치 동일 — 3090 24GB 대응).
        # 짧은 시퀀스는 전 head 한 번에 (오버헤드 없음).
        chunk = n_heads if (q_len <= CTX.chunk_seq_threshold) else CTX.head_chunk
        do_inject = CTX.enabled and id(self) in CTX.layer_module_ids
        layer_idx = getattr(self, "layer_idx", None)
        rec_rows = []
        outs = []
        for h0 in range(0, n_heads, chunk):
            q = query_states[:, h0:h0 + chunk]
            k = key_states[:, h0:h0 + chunk]
            v = value_states[:, h0:h0 + chunk]
            aw = torch.matmul(q, k.transpose(2, 3)) * scale
            if causal_mask is not None:
                aw = aw + causal_mask                      # (1,1,L,S) 브로드캐스트
            if q.dtype == torch.float16:
                aw = torch.where(torch.isinf(aw), torch.zeros_like(aw), aw)
            aw = nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)

            # ==== AVCD 삽입점 (원본과의 유일한 차이) ====
            if do_inject and q_len > 1 and layer_idx is not None:
                if CTX.recording:
                    rec_rows.append(aw[0, :, -1, :].detach().to(torch.float32).cpu())
                if CTX.span_mask is not None and layer_idx < CTX.num_layers - 1:
                    if CTX.span_mask.shape[0] != S:
                        raise RuntimeError(
                            f"AVCD span_mask 길이({CTX.span_mask.shape[0]}) != attention S({S}) — "
                            f"span 계산이 실제 시퀀스와 어긋남 (runbook T3)")
                    # 메모리: fp32 복사 1개만 유지 — 원본 aw 즉시 해제, in-place 마스킹 (수식 동일)
                    aw_f32 = aw[0].to(torch.float32)
                    span = CTX.span_mask.to(aw_f32.device)
                    del aw
                    aw = mask_attention_rows(
                        aw_f32, span, CTX.threshold, inplace=True,
                    ).unsqueeze(0).to(q.dtype)
                    del aw_f32
            # ============================================

            aw = nn.functional.dropout(aw, p=self.attention_dropout, training=self.training)
            outs.append(torch.matmul(aw, v))
            del aw
        attn_output = torch.cat(outs, dim=1)
        if do_inject and CTX.recording and rec_rows and layer_idx is not None:
            CTX.recorded[layer_idx] = torch.cat(rec_rows, dim=0)   # (H, S)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        attn_weights = None                    # 기록은 CTX로 — output_attentions 미지원(불필요)
        return attn_output, attn_weights, past_key_value

    forward.__avcd_patched__ = True
    return forward


def patch_qwen25_omni(model) -> AVCDPatchContext:
    """Qwen2.5-Omni thinker text decoder에 AVCD 패치 적용. 반환된 CTX로 제어.

    transformers 버전에 따라 두 경로 중 하나를 자동 선택:
    - 신식 (≥4.54 계열, 로컬 4.57.6 검증): 모듈 함수 eager_attention_forward 래핑
    - 구식 (서버 4.52.3 검증): Qwen2_5OmniAttention.forward 클래스 교체 (verbatim+삽입)
    """
    import transformers
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as m

    # mixed 로드(인코더 flash + thinker eager 스왑)를 허용 — thinker text 기준으로만 검사
    thinker_impl = getattr(model.thinker.model.config, "_attn_implementation", None)
    assert thinker_impl == "eager", (
        f"AVCD는 thinker text attention이 eager여야 합니다 (현재: {thinker_impl}) — "
        "어댑터의 _make_thinker_eager 적용 여부 확인")
    layers = model.thinker.model.layers

    if hasattr(m, "eager_attention_forward"):
        return install_patch(m, [l.self_attn for l in layers])

    # 구식 경로 (4.52.x)
    assert hasattr(m, "Qwen2_5OmniAttention"), (
        f"transformers {transformers.__version__}: 신식 함수도 구식 클래스도 없음 — "
        "runbook T4 절차로 패치 지점 재확인 필요")
    attn_cls = m.Qwen2_5OmniAttention
    for l in layers:
        assert isinstance(l.self_attn, attn_cls), (
            f"thinker layer attention이 {type(l.self_attn).__name__} — eager 로드 확인 필요")
    CTX.layer_module_ids = frozenset(id(l.self_attn) for l in layers)
    CTX.num_layers = len(layers)
    if not getattr(attn_cls.forward, "__avcd_patched__", False):
        attn_cls.forward = _legacy_forward_factory(m)
        logger.info("Qwen2.5-Omni(구식 4.52.x) attention 클래스 패치 완료 (%d layers)", CTX.num_layers)
    return CTX
