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

        layer_idx = getattr(module, "layer_idx", None)
        q_len = attn_weights.shape[-2]

        if q_len > 1:  # prefill(전체 시퀀스)에서만 기록/마스킹 (공식 AVCD와 동일 조건)
            if CTX.recording and layer_idx is not None:
                CTX.recorded[layer_idx] = attn_weights[0, :, -1, :].detach().to(torch.float32).cpu()
            if (CTX.span_mask is not None and layer_idx is not None
                    and layer_idx < CTX.num_layers - 1):        # 마지막 layer 제외
                S = attn_weights.shape[-1]
                if CTX.span_mask.shape[0] != S:
                    raise RuntimeError(
                        f"AVCD span_mask 길이({CTX.span_mask.shape[0]}) != attention S({S}) — "
                        f"span 계산이 실제 시퀀스와 어긋남 (S1에서 점검)")
                attn_weights = mask_attention_rows(
                    attn_weights[0], CTX.span_mask.to(attn_weights.device), CTX.threshold
                ).unsqueeze(0)

        attn_weights = attn_weights.to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    eager_with_avcd.__avcd_patched__ = True
    return eager_with_avcd


def patch_qwen25_omni(model) -> AVCDPatchContext:
    """Qwen2.5-Omni 모델의 thinker text decoder에 AVCD 패치 적용. 반환된 CTX로 제어."""
    import transformers
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as m

    assert hasattr(m, "eager_attention_forward"), (
        f"transformers {transformers.__version__}: eager_attention_forward가 없습니다 — "
        "패치 지점 변경 필요 (runbook '패치 검증' 항목 참조)")
    assert model.config._attn_implementation == "eager", (
        "AVCD는 attn_implementation='eager' 로드가 필수입니다")

    layers = model.thinker.model.layers
    CTX.layer_module_ids = frozenset(id(l.self_attn) for l in layers)
    CTX.num_layers = len(layers)

    if not getattr(m.eager_attention_forward, "__avcd_patched__", False):
        m.eager_attention_forward = _wrapped_eager(m.eager_attention_forward)
        logger.info("Qwen2.5-Omni eager attention 패치 완료 (%d layers)", CTX.num_layers)
    return CTX
