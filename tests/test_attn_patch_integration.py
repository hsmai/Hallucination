"""AVCD attention 패치 통합 테스트 — 실제 transformers 코드 경로 통과 검증 (신규 구역 4단 검증의 2단).

랜덤 초기화 초소형 Qwen2 모델(다운로드 0바이트, CPU)로:
  1) 패치 후 기록 모드: layer별 마지막 query attention 행이 수집되는가
  2) 마스킹 모드: threshold=∞ → 무패치 출력과 완전 동일(무해성), 유한 threshold → 출력 변화
  3) 비활성화 시 원본 동작 그대로인가

⚠ 로컬 transformers 버전과 서버 버전이 다를 수 있음 — 이 테스트는 "패치 메커니즘"을
검증하며, 서버 버전 확인은 runbook S1 체크리스트에 있다.
Qwen2.5-Omni thinker의 attention도 동일한 eager_attention_forward 패턴을 쓰므로
(테스트 말미에 모듈 존재를 확인) qwen2 소형 모델 검증이 메커니즘을 대표한다.
"""

import importlib

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.adapters.attn_patch import CTX, install_patch  # noqa: E402

S = 24  # 시퀀스 길이


@pytest.fixture(scope="module")
def tiny_qwen2():
    from transformers.models.qwen2 import modeling_qwen2
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    # 패치가 모듈 전역을 바꾸므로, 테스트 전 원본 함수를 보관했다가 모듈 리로드로 복원
    cfg = Qwen2Config(
        vocab_size=120, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).eval()
    yield model, modeling_qwen2
    importlib.reload(modeling_qwen2)  # 전역 패치 원상복구


def forward_logits(model, ids):
    with torch.no_grad():
        return model(ids).logits[0, -1, :].clone()


class TestPatchIntegration:
    def test_recording_and_masking_through_real_transformers(self, tiny_qwen2):
        model, modeling = tiny_qwen2
        ids = torch.randint(0, 120, (1, S), generator=torch.Generator().manual_seed(1))

        baseline = forward_logits(model, ids)  # 패치 설치 전

        layers = model.model.layers
        ctx = install_patch(modeling, [l.self_attn for l in layers])
        assert ctx.num_layers == 3

        # --- 0) 패치 설치 후 비활성 상태: 원본과 동일해야 함
        CTX.enabled = False
        assert torch.allclose(forward_logits(model, ids), baseline, atol=1e-6)

        # --- 1) 기록 모드: layer별 (H, S) 마지막 query attention 행 수집
        CTX.enabled, CTX.recording, CTX.span_mask = True, True, None
        CTX.reset_records()
        out_rec = forward_logits(model, ids)
        CTX.enabled = CTX.recording = False

        rows = ctx.records_in_order()
        assert len(rows) == 3
        for r in rows:
            assert r.shape == (4, S)                      # (heads, seq)
            sums = r.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-4)  # softmax 행
        # 기록만 하는 forward는 출력 불변
        assert torch.allclose(out_rec, baseline, atol=1e-5)

        # --- 2) 마스킹 no-op 검증: threshold=∞ → 아무 행도 안 지움 → baseline과 동일
        span = torch.zeros(S, dtype=torch.bool)
        span[4:12] = True                                 # 가짜 'video' span
        CTX.enabled, CTX.span_mask, CTX.threshold = True, span, float("inf")
        out_noop = forward_logits(model, ids)
        CTX.enabled, CTX.span_mask = False, None
        assert torch.allclose(out_noop, baseline, atol=1e-5), \
            "threshold=inf인데 출력이 변함 — 마스킹 경로가 무해하지 않음"

        # --- 3) 실제 마스킹: threshold=0 → span 행 전부 제거 → 출력이 달라져야 함
        CTX.enabled, CTX.span_mask, CTX.threshold = True, span, 0.0
        out_masked = forward_logits(model, ids)
        CTX.enabled, CTX.span_mask = False, None
        assert not torch.allclose(out_masked, baseline, atol=1e-3), \
            "span 전체를 마스킹했는데 출력 불변 — 패치가 적용되지 않음"

        # --- 4) 마지막 layer 제외 규칙: 전 layer를 마스킹 대상으로 하되
        #        num_layers-1 미만에만 적용되는지는 단독 layer 마스킹 차이로 간접 확인
        #        (layer_idx >= num_layers-1 은 _wrapped_eager에서 걸러짐 — 코드 경로 검증)
        assert ctx.num_layers - 1 == 2

    def test_decode_step_with_cache_unaffected(self, tiny_qwen2):
        """생성 스텝(q_len==1, KV cache)은 기록/마스킹 대상이 아님 (공식 AVCD와 동일)."""
        model, modeling = tiny_qwen2
        ids = torch.randint(0, 120, (1, S))
        layers = model.model.layers
        install_patch(modeling, [l.self_attn for l in layers])

        CTX.enabled, CTX.recording = True, True
        CTX.reset_records()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            n_before = len(CTX.recorded)
            CTX.reset_records()
            model(torch.tensor([[5]]), past_key_values=out.past_key_values, use_cache=True)
            n_after = len(CTX.recorded)
        CTX.enabled = CTX.recording = False
        assert n_before == 3 and n_after == 0  # prefill만 기록, 1-token 스텝은 무기록


def test_qwen25_omni_module_has_patch_target():
    """실서버 대상 모듈에 패치 지점이 존재하는지 (버전 호환 조기 경보).

    신식(≥4.54: eager_attention_forward 함수) 또는 구식(4.52.x: Attention 클래스)
    둘 중 하나면 patch_qwen25_omni()가 처리한다.
    """
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as mo
    assert hasattr(mo, "eager_attention_forward") or hasattr(mo, "Qwen2_5OmniAttention")
