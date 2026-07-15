"""AVCD 핵심 수학 — 모델 무관 순수 함수 (CPU 더미 텐서로 단위 테스트 가능).

공식 코드 대조 근거: third_party/AVCD/videollama2/model/qwen.py
- dominance: L1176–1223 (마지막 query 토큰 attention, head 평균, span 합/span 길이, 마지막 layer 제외)
- threshold: L1212–1213 (layer 누적 평균의 0.5-quantile → head 평균 → 스칼라)
- masking:  L732–779 (span에 속한 "query 행" 중 last-query attention이 threshold 초과인
  행을 통째로 zeroing 후 행 재정규화. key 열이 아니라 query 행을 지운다 — 코드 그대로 재현)

주의: AVCD 논문 서술("top P% 토큰 zeroing")보다 코드가 구체적이며, 우리는 코드를 따른다
(작업 원칙: 논문·코드 충돌 시 코드 우선 + 보고 — docs/code_analysis.md §2.4).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

MODALITY_KEYS = ("video", "audio", "language")

# mask_spec 문자 → span 키
SPEC_CHARS = {"V": "video", "A": "audio", "L": "language"}


# ---------------------------------------------------------------- span 계산

def videollama2_spans(seq_len: int, head_len: int, n_video: int, n_audio: int) -> Dict[str, torch.Tensor]:
    """VideoLLaMA2-AV: [head(system) | video | audio | language(question)] 연속 배치.

    AVCD 공식 코드는 head=14, n_video=676, n_audio=1496을 하드코딩했으나 우리는 동적으로 받는다
    (placeholder 토큰 위치 + 인코더 출력 길이로 어댑터가 계산). language는 AVCD와 동일하게
    head를 제외한 꼬리(질문+generation prompt) 구간만 포함한다.
    """
    if head_len + n_video + n_audio > seq_len:
        raise ValueError(f"span 합({head_len}+{n_video}+{n_audio})이 seq_len({seq_len}) 초과")
    spans = {k: torch.zeros(seq_len, dtype=torch.bool) for k in MODALITY_KEYS}
    spans["video"][head_len: head_len + n_video] = True
    spans["audio"][head_len + n_video: head_len + n_video + n_audio] = True
    spans["language"][head_len + n_video + n_audio:] = True
    return spans


def qwen_spans(input_ids: Sequence[int], audio_token_id: int, video_token_id: int) -> Dict[str, torch.Tensor]:
    """Qwen2.5-Omni: <|AUDIO|>/<|VIDEO|> pad 토큰이 input_ids에 남아 있어 id로 span을 찾는다.

    use_audio_in_video=True면 audio/video 토큰이 시간순 interleave → 연속 구간이 아니므로
    boolean mask로 처리. language span은 AVCD의 관례(head/system 제외, 질문 꼬리만)를 따라
    '마지막 미디어 토큰 이후'의 텍스트 토큰으로 정의한다. (서버 S1에서 실제 시퀀스로 검증)
    """
    ids = torch.as_tensor(list(input_ids), dtype=torch.long)
    seq_len = ids.numel()
    video = ids == video_token_id
    audio = ids == audio_token_id
    if not (video.any() or audio.any()):
        raise ValueError("input_ids에 audio/video placeholder 토큰이 없습니다 — token id 확인 필요")
    media = video | audio
    last_media = int(torch.nonzero(media).max())
    language = torch.zeros(seq_len, dtype=torch.bool)
    language[last_media + 1:] = True
    return {"video": video, "audio": audio, "language": language}


def spec_to_mask(spec: str, spans: Dict[str, torch.Tensor]) -> torch.Tensor:
    """"VA"/"LV"/"A" 등 mask_spec → 해당 modality들의 합집합 boolean mask [seq]."""
    if not spec or any(c not in SPEC_CHARS for c in spec):
        raise ValueError(f"잘못된 mask_spec: {spec!r} (허용 문자: V,A,L)")
    m = torch.zeros_like(next(iter(spans.values())))
    for c in spec:
        m |= spans[SPEC_CHARS[c]]
    return m


# ---------------------------------------------------------------- dominance / threshold

def dominance_and_threshold(
    last_query_attn_per_layer: List[torch.Tensor],   # layer별 (H, S): 마지막 query의 attention 행
    spans: Dict[str, torch.Tensor],
    exclude_last_layer: bool = True,
) -> Tuple[List[Tuple[str, float]], float]:
    """AVCD dominance 순위와 masking threshold.

    - dominance(modality) = mean_layers[ sum_span(head-mean attn) / |span| ], 마지막 layer 제외
    - threshold = quantile_{0.5, key dim}( mean_layers(head별 attn 누적) ) 의 head 평균 (스칼라)
      ⚠ 공식 코드는 평균 분모를 len(전체 layer)로 쓰는 사소한 버그가 있으나 순위에 무영향
        (docs/code_analysis.md §2.2) — 우리는 제외 후 layer 수로 나눈다. 순위·중앙값 모두 불변.
    """
    n_layers = len(last_query_attn_per_layer)
    if n_layers == 0:
        raise ValueError("layer attention이 비어 있습니다")
    use = last_query_attn_per_layer[:-1] if (exclude_last_layer and n_layers > 1) \
        else last_query_attn_per_layer

    scores = {k: 0.0 for k in MODALITY_KEYS}
    acc = torch.zeros_like(use[0])                    # (H, S) 누적
    for attn in use:
        if attn.shape != use[0].shape:
            raise ValueError("layer 간 attention shape 불일치")
        acc += attn
        head_mean = attn.mean(dim=0)                  # (S,)
        for k in MODALITY_KEYS:
            span = spans[k]
            size = int(span.sum())
            if size > 0:
                scores[k] += float(head_mean[span].sum()) / size
    for k in scores:
        scores[k] = abs(scores[k] / len(use))

    acc = acc / len(use)                              # (H, S)
    threshold = float(torch.quantile(acc, 0.5, dim=-1).mean())

    dominance = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dominance, threshold


# ---------------------------------------------------------------- attentive masking

def mask_attention_rows(
    attn: torch.Tensor,          # (H, L, S) softmax 후 attention (prefill: L == S)
    span_mask: torch.Tensor,     # (S,) bool — 마스킹 대상 modality 합집합
    threshold: float,
    renorm_eps: float = 1e-6,
) -> torch.Tensor:
    """AVCD attentive masking (공식 코드 L732–779 재현).

    span에 속한 위치 p의 "query 행" attn[h, p, :]를, 마지막 query가 p를 보는 강도
    attn[h, -1, p]가 threshold를 초과하는 (h, p)에 대해 통째로 0으로 만들고 행 재정규화.
    (전부 0이 된 행은 clamp에 의해 0으로 유지 → 해당 토큰의 attention 출력 제거)
    """
    if attn.dim() != 3:
        raise ValueError(f"attn은 (H, L, S)여야 합니다: {tuple(attn.shape)}")
    H, L, S = attn.shape
    if L != S:
        # 마스킹은 prefill(전체 시퀀스 forward)에서만 적용 (공식 코드: size(2)!=1 조건)
        raise ValueError("row masking은 L==S(prefill)에서만 정의됨")
    if span_mask.shape != (S,):
        raise ValueError("span_mask shape 불일치")

    last_query = attn[:, -1, :]                       # (H, S)
    keep = (last_query <= threshold)                  # True=유지, False=제거 대상
    row_mask = torch.ones_like(attn)
    # span 위치의 query 행에 keep 여부 브로드캐스트 (공식 코드의 modality_mask[:, span, :] = av_mask)
    span_idx = torch.nonzero(span_mask).squeeze(-1)
    row_mask[:, span_idx, :] = keep[:, span_idx].unsqueeze(-1).to(attn.dtype)

    masked = attn * row_mask
    denom = masked.sum(dim=-1, keepdim=True).clamp(min=renorm_eps)
    return masked / denom
