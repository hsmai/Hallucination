"""AVCD attention 수학 (src/adapters/common.py) CPU 검증 — 소형 더미 텐서.

hidden/vocab 크기와 무관한 순수 attention 조작이므로 seq 50, head 4 수준으로 검증.
공식 코드(third_party/AVCD/videollama2/model/qwen.py)의 동작을 기준으로 한다.
"""

import pytest
import torch

from src.adapters.common import (
    dominance_and_threshold,
    mask_attention_rows,
    qwen_spans,
    spec_to_mask,
    videollama2_spans,
)

H, S = 4, 50  # heads, seq


def softmax_rows(x):
    return torch.softmax(x, dim=-1)


@pytest.fixture
def spans():
    # [head 5 | video 20 | audio 15 | language 10]
    return videollama2_spans(seq_len=S, head_len=5, n_video=20, n_audio=15)


# ---------------------------------------------------------------- span 계산

class TestSpans:
    def test_videollama2_layout(self, spans):
        assert int(spans["video"].sum()) == 20
        assert int(spans["audio"].sum()) == 15
        assert int(spans["language"].sum()) == 10          # head 5는 어느 span에도 없음
        assert not bool(spans["video"][:5].any())
        assert bool(spans["video"][5]) and bool(spans["audio"][25]) and bool(spans["language"][40])
        # 서로 배타
        assert not bool((spans["video"] & spans["audio"]).any())

    def test_videollama2_overflow_raises(self):
        with pytest.raises(ValueError):
            videollama2_spans(seq_len=30, head_len=5, n_video=20, n_audio=15)

    def test_qwen_interleaved(self):
        AUD, VID, TXT = 7, 8, 1
        # interleave: [txt txt VID AUD VID AUD txt txt txt] — Qwen use_audio_in_video 형태
        ids = [TXT, TXT, VID, AUD, VID, AUD, TXT, TXT, TXT]
        sp = qwen_spans(ids, audio_token_id=AUD, video_token_id=VID)
        assert sp["video"].tolist() == [False, False, True, False, True, False, False, False, False]
        assert sp["audio"].tolist() == [False, False, False, True, False, True, False, False, False]
        # language = 마지막 미디어(idx5) 이후 텍스트만 (head 텍스트 제외)
        assert sp["language"].tolist() == [False] * 6 + [True] * 3

    def test_qwen_no_media_raises(self):
        with pytest.raises(ValueError):
            qwen_spans([1, 2, 3], audio_token_id=7, video_token_id=8)

    def test_spec_to_mask_union(self, spans):
        m = spec_to_mask("VA", spans)
        assert int(m.sum()) == 35
        m2 = spec_to_mask("LV", spans)
        assert int(m2.sum()) == 30
        with pytest.raises(ValueError):
            spec_to_mask("X", spans)


# ---------------------------------------------------------------- dominance / threshold

class TestDominance:
    def make_layers(self, n_layers, boost_span=None, spans=None, boost=10.0):
        """마지막 query 행 (H,S)를 layer별 생성. boost_span에 attention 몰아주기."""
        layers = []
        for i in range(n_layers):
            g = torch.Generator().manual_seed(i)
            raw = torch.rand(H, S, generator=g)
            if boost_span is not None:
                raw[:, spans[boost_span]] += boost
            layers.append(softmax_rows(raw))
        return layers

    def test_argmax_dominant(self, spans):
        for target in ("video", "audio", "language"):
            layers = self.make_layers(6, boost_span=target, spans=spans)
            dom, _ = dominance_and_threshold(layers, spans)
            assert dom[0][0] == target, f"{target} 몰아줬는데 {dom[0][0]}가 dominant"

    def test_span_size_normalization(self, spans):
        """전 위치 균등 attention이면 span 길이 정규화 덕에 세 modality 점수 동일."""
        layers = [torch.full((H, S), 1.0 / S) for _ in range(4)]
        dom, _ = dominance_and_threshold(layers, spans)
        scores = dict(dom)
        assert scores["video"] == pytest.approx(scores["audio"], rel=1e-6)
        assert scores["video"] == pytest.approx(scores["language"], rel=1e-6)

    def test_last_layer_excluded(self, spans):
        """마지막 layer에만 video를 몰아줘도 dominance에 반영되지 않아야 한다."""
        layers = self.make_layers(5)  # 균등 랜덤
        last = torch.rand(H, S)
        last[:, spans["video"]] += 100.0
        layers.append(softmax_rows(last))
        dom_ex, _ = dominance_and_threshold(layers, spans, exclude_last_layer=True)
        dom_in, _ = dominance_and_threshold(layers, spans, exclude_last_layer=False)
        assert dom_in[0][0] == "video"
        # 제외 시에는 랜덤 균등이라 video가 압도하지 않음 (몰아준 layer가 무시됨)
        scores_ex = dict(dom_ex)
        assert scores_ex["video"] < dict(dom_in)["video"]

    def test_threshold_is_median_head_mean(self, spans):
        """균등 attention이면 threshold == 1/S (모든 위치 동일 → 중앙값 = 값 자체)."""
        layers = [torch.full((H, S), 1.0 / S) for _ in range(4)]
        _, th = dominance_and_threshold(layers, spans)
        assert th == pytest.approx(1.0 / S, rel=1e-5)


# ---------------------------------------------------------------- attentive masking

class TestMaskRows:
    def test_top50_rows_zeroed_and_renormalized(self, spans):
        g = torch.Generator().manual_seed(42)
        attn = softmax_rows(torch.randn(H, S, S, generator=g))
        span = spans["video"]
        # threshold = 마지막 query의 video 위치 attention 중앙값 → 절반가량 제거 기대
        last_q_video = attn[:, -1, :][:, span]
        threshold = float(last_q_video.median())

        out = mask_attention_rows(attn, span, threshold)

        # 1) 제거된 행: span 위치 p 중 last-query attn > threshold인 (h,p) — 행 전체 0
        last_query = attn[:, -1, :]
        removed = 0
        for h in range(H):
            for p in torch.nonzero(span).squeeze(-1).tolist():
                if float(last_query[h, p]) > threshold:
                    assert torch.all(out[h, p, :] == 0), f"행 ({h},{p})가 0이 아님"
                    removed += 1
                else:
                    assert float(out[h, p, :].sum()) == pytest.approx(1.0, abs=1e-4)
        n_span = int(span.sum()) * H
        assert 0.3 <= removed / n_span <= 0.7  # 중앙값 threshold → 대략 절반

        # 2) span 밖 행은 재정규화만 (원래 softmax 행이라 그대로 합=1)
        outside = torch.nonzero(~span).squeeze(-1)
        for p in outside.tolist()[:5]:
            assert float(out[0, p, :].sum()) == pytest.approx(1.0, abs=1e-4)

    def test_shape_and_errors(self, spans):
        attn = softmax_rows(torch.randn(H, S, S))
        out = mask_attention_rows(attn, spans["audio"], 0.5)
        assert out.shape == (H, S, S)
        with pytest.raises(ValueError):
            mask_attention_rows(attn[:, :10, :], spans["audio"], 0.5)  # L != S
        with pytest.raises(ValueError):
            mask_attention_rows(attn, spans["audio"][:20], 0.5)       # span 길이 불일치

    def test_threshold_extremes(self, spans):
        attn = softmax_rows(torch.randn(H, S, S))
        span = spans["video"]
        # threshold가 매우 크면 아무것도 제거 안 됨 → 원본과 동일
        out = mask_attention_rows(attn, span, threshold=1e9)
        assert torch.allclose(out, attn, atol=1e-6)
        # threshold=0이면 span 행 전부 제거 (softmax 값은 전부 > 0)
        out0 = mask_attention_rows(attn, span, threshold=0.0)
        span_idx = torch.nonzero(span).squeeze(-1)
        assert torch.all(out0[:, span_idx, :] == 0)

    def test_inplace_equivalence_and_purity(self, spans):
        """2026-07-20 OOM 교정(행-스케일 in-place)이 구현 전 수식과 완전 동일함을 검증.

        - 참조 구현: 교정 전 코드 그대로 (ones_like row_mask → masked/denom)
        - inplace=False(기본)는 입력을 변경하지 않아야 함 (기존 계약 유지)
        - inplace=True는 결과 동일 + 입력 텐서를 제자리 수정
        """
        def reference(attn, span_mask, threshold, eps=1e-6):
            last_query = attn[:, -1, :]
            keep = (last_query <= threshold)
            row_mask = torch.ones_like(attn)
            span_idx = torch.nonzero(span_mask).squeeze(-1)
            row_mask[:, span_idx, :] = keep[:, span_idx].unsqueeze(-1).to(attn.dtype)
            masked = attn * row_mask
            denom = masked.sum(dim=-1, keepdim=True).clamp(min=eps)
            return masked / denom

        for span_key in ("audio", "video"):
            span = spans[span_key]
            attn = softmax_rows(torch.randn(H, S, S))
            # threshold를 중앙값 근처로 잡아 실제로 절반쯤 제거되는 경로를 태운다
            thr = float(attn[:, -1, :].median())
            ref = reference(attn.clone(), span, thr)

            orig = attn.clone()
            out = mask_attention_rows(attn, span, thr)            # 기본: 비파괴
            assert torch.allclose(out, ref, atol=1e-6)
            assert torch.equal(attn, orig), "inplace=False가 입력을 변경함"

            buf = attn.clone()
            out_ip = mask_attention_rows(buf, span, thr, inplace=True)
            assert torch.allclose(out_ip, ref, atol=1e-6)
            assert out_ip.data_ptr() == buf.data_ptr(), "inplace=True인데 새 텐서 반환"

    def test_qblock_mode_equals_full(self, spans):
        """q-block 모드(keep 사전계산 + row_offset 블록 처리)가 전체 처리와 동일한지.

        2026-07-20 초장문 OOM 대응: attn_patch가 초장문 prefill에서 행 블록으로
        나눠 호출하는 경로의 수치 동등성을 함수 레벨에서 검증한다.
        """
        span = spans["video"]
        attn = softmax_rows(torch.randn(H, S, S))
        thr = float(attn[:, -1, :].median())
        full = mask_attention_rows(attn, span, thr)

        keep = (attn[:, -1, :] <= thr)
        for block in (5, 7, S):   # 비약수 블록 크기 포함
            parts = []
            for r0 in range(0, S, block):
                r1 = min(r0 + block, S)
                parts.append(mask_attention_rows(
                    attn[:, r0:r1, :].clone(), span, thr,
                    inplace=True, keep=keep, row_offset=r0))
            got = torch.cat(parts, dim=1)
            assert torch.allclose(got, full, atol=1e-6), f"block={block} 불일치"
