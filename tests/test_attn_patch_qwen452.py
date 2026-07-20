"""구식(4.52.x) Qwen2_5OmniAttention 클래스 패치 검증 — 서버 qwen-omni env 전용.

로컬(신식 transformers)에서는 자동 skip. 서버에서:
    conda run -n qwen-omni python -m pytest tests/test_attn_patch_qwen452.py -q

초소형 attention 모듈(랜덤 초기화, 다운로드 0바이트)로:
  1) 패치 후 비활성 상태 == 원본 출력 (무해성)
  2) 기록 모드: (H, S) softmax 행 수집 + 출력 불변
  3) threshold=∞ no-op / threshold=0 출력 변화
"""

import pytest
import torch

pytest.importorskip("transformers")
m = pytest.importorskip(
    "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni",
    reason="qwen2_5_omni 미지원 transformers (videollama2 env 등) — skip")

if hasattr(m, "eager_attention_forward") or not hasattr(m, "Qwen2_5OmniAttention"):
    pytest.skip("구식(4.52.x) 클래스 경로가 아님 — 서버 qwen-omni env에서만 실행", allow_module_level=True)

from src.adapters.attn_patch import CTX, _legacy_forward_factory  # noqa: E402

H, KV, D, S = 4, 2, 16, 24  # heads, kv_heads, head_dim, seq


class _TinyCfg:
    hidden_size = H * D
    num_attention_heads = H
    num_key_value_heads = KV
    head_dim = D
    attention_dropout = 0.0
    rope_theta = 10000.0
    rope_scaling = {"mrope_section": [2, 3, 3], "rope_type": "default", "type": "default"}
    max_position_embeddings = 128


@pytest.fixture(scope="module")
def attn_and_inputs():
    torch.manual_seed(0)
    cfg = _TinyCfg()
    attn = m.Qwen2_5OmniAttention(cfg, layer_idx=0).eval()

    hidden = torch.randn(1, S, cfg.hidden_size)
    rotary = m.Qwen2_5OmniRotaryEmbedding(config=cfg)
    position_ids = torch.arange(S).view(1, 1, S).expand(3, 1, S)  # mrope (3, B, S)
    cos, sin = rotary(hidden, position_ids)
    causal = torch.full((1, 1, S, S), float("-inf")).triu(1)
    return attn, dict(hidden_states=hidden, attention_mask=causal,
                      position_embeddings=(cos, sin))


def run(attn, inputs):
    with torch.no_grad():
        out, _, _ = attn.forward(**inputs)
    return out


class TestLegacyPatch:
    def test_full_cycle(self, attn_and_inputs):
        attn, inputs = attn_and_inputs
        baseline = run(attn, inputs)

        orig_forward = type(attn).forward
        type(attn).forward = _legacy_forward_factory(m)
        CTX.layer_module_ids = frozenset([id(attn)])
        CTX.num_layers = 2  # layer_idx 0 < 1 → 마스킹 적용 대상
        try:
            # 0) 비활성: 원본과 동일
            CTX.enabled = False
            assert torch.allclose(run(attn, inputs), baseline, atol=1e-5)

            # 1) 기록 모드
            CTX.enabled, CTX.recording, CTX.span_mask = True, True, None
            CTX.reset_records()
            out_rec = run(attn, inputs)
            CTX.recording = False
            rows = CTX.records_in_order()
            assert len(rows) == 1 and rows[0].shape == (H, S)
            assert torch.allclose(rows[0].sum(-1), torch.ones(H), atol=1e-4)
            assert torch.allclose(out_rec, baseline, atol=1e-5)

            # 2) threshold=∞ → no-op
            span = torch.zeros(S, dtype=torch.bool); span[3:10] = True
            CTX.span_mask, CTX.threshold = span, float("inf")
            assert torch.allclose(run(attn, inputs), baseline, atol=1e-5)

            # 3) threshold=0 → span 행 제거 → 출력 변화
            CTX.threshold = 0.0
            assert not torch.allclose(run(attn, inputs), baseline, atol=1e-3)
        finally:
            CTX.enabled, CTX.recording, CTX.span_mask = False, False, None
            type(attn).forward = orig_forward


class TestLegacyQBlock:
    def test_qblock_equals_nonqblock(self, attn_and_inputs):
        """q-block 경로(초장문용)가 비블록 경로와 출력·기록 모두 동일한지 (서버 env)."""
        attn, inputs = attn_and_inputs
        orig_forward = type(attn).forward
        type(attn).forward = _legacy_forward_factory(m)
        CTX.layer_module_ids = frozenset([id(attn)])
        CTX.num_layers = 2
        saved = (CTX.qblock_seq_threshold, CTX.q_block)
        span = torch.zeros(S, dtype=torch.bool); span[3:10] = True
        try:
            for mode in ("record", "mask"):
                CTX.enabled = True
                CTX.recording = (mode == "record")
                CTX.span_mask = span if mode == "mask" else None
                CTX.threshold = 0.02 if mode == "mask" else 0.0

                CTX.qblock_seq_threshold, CTX.q_block = 10 ** 9, 4096  # 비블록
                CTX.reset_records()
                ref = run(attn, inputs)
                ref_rows = CTX.records_in_order()

                CTX.qblock_seq_threshold, CTX.q_block = 8, 7          # 블록 (비약수)
                CTX.reset_records()
                got = run(attn, inputs)
                got_rows = CTX.records_in_order()

                assert torch.allclose(got, ref, atol=1e-5), f"{mode}: 출력 불일치"
                assert len(ref_rows) == len(got_rows)
                for a, b in zip(ref_rows, got_rows):
                    assert torch.allclose(a, b, atol=1e-6), f"{mode}: 기록 행 불일치"
        finally:
            CTX.qblock_seq_threshold, CTX.q_block = saved
            CTX.enabled, CTX.recording, CTX.span_mask = False, False, None
            type(attn).forward = orig_forward
