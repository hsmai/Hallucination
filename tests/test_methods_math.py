"""방법 수식 CPU 검증 (소형 더미 텐서) — Phase L4.

검증 목표 (claude_code_master_prompt.md L4-4):
- MAD: weight softmax → 4-branch 가중합이 논문 Eq.9 전개와 일치
- VCD-ext: 계수 (1+3α, −α, −α, −α)
- AVCD: 결합식 계수, EAD 분기, plausibility 마스킹
"""

import math

import pytest
import torch

from src.methods.mad import MAD
from src.methods.avcd import AVCD

VOCAB = 100


def rand_logits(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, generator=g)


# ---------------------------------------------------------------- MAD

class TestMADWeights:
    def test_eq9_expansion_matches_branch_weights(self):
        """branch 가중합 == MAD 논문 Eq.9 원식 전개 (임의 확률·γ에서)."""
        g = torch.Generator().manual_seed(0)
        for trial in range(20):
            gamma = float(torch.rand(1, generator=g)) * 4
            z = torch.randn(3, generator=g)
            p_audio, p_video, p_both = torch.softmax(z, dim=0).tolist()

            l_va, l_v, l_a, l_t = (rand_logits(trial * 4 + i) for i in range(4))
            # 브랜치 매핑: l_ṽaq(비디오 제거)=l_a, l_vãq(오디오 제거)=l_v, l_ṽãq=l_t
            eq9 = ((1 + gamma * p_both) * l_va - gamma * p_both * l_a
                   + (1 + gamma * p_both) * l_va - gamma * p_both * l_v
                   + (1 + gamma * p_video) * l_v - gamma * p_video * l_t
                   + (1 + gamma * p_audio) * l_a - gamma * p_audio * l_t)

            w = MAD.branch_weights(gamma, p_audio, p_video, p_both)
            fused = w[0] * l_va + w[1] * l_v + w[2] * l_a + w[3] * l_t
            assert torch.allclose(fused, eq9, atol=1e-5), f"trial {trial} 불일치"

    def test_weights_sum_invariant(self):
        """가중치 총합은 γ·확률과 무관하게 항상 4 (docs/paper_settings.md §1)."""
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            gamma = float(torch.rand(1, generator=g)) * 5
            p = torch.softmax(torch.randn(3, generator=g), dim=0).tolist()
            w = MAD.branch_weights(gamma, *p)
            assert math.isclose(sum(w), 4.0, abs_tol=1e-6)

    def test_repo_code_form_equivalence(self):
        """MAD repo 코드 표기(alpha_*)와 우리 branch_weights가 동일."""
        gamma, p_audio, p_video, p_both = 2.5, 0.2, 0.5, 0.3
        alpha_av = 2 * p_both * gamma
        alpha_v = (p_both - p_video) * gamma
        alpha_a = (p_both - p_audio) * gamma
        alpha_t = (p_video + p_audio) * gamma
        repo = [2 + alpha_av, 1 - alpha_v, 1 - alpha_a, -alpha_t]
        ours = MAD.branch_weights(gamma, p_audio, p_video, p_both)
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(repo, ours))


# ---------------------------------------------------------------- VCD-ext

class TestVCDExt:
    def test_coefficients(self):
        """fused == (1+3α)·l_va − α·(l_v + l_a + l_t)."""
        alpha = 0.5
        l = [rand_logits(i) for i in range(4)]
        weights = [1 + 3 * alpha, -alpha, -alpha, -alpha]
        fused = torch.stack([w * x for w, x in zip(weights, l)]).sum(0)
        direct = (1 + 3 * alpha) * l[0] - alpha * (l[1] + l[2] + l[3])
        assert torch.allclose(fused, direct, atol=1e-6)

    def test_alpha_zero_is_base(self):
        """α=0이면 원본 logits와 동일 (greedy base로 환원)."""
        l = [rand_logits(i + 10) for i in range(4)]
        weights = [1.0, 0.0, 0.0, 0.0]
        fused = torch.stack([w * x for w, x in zip(weights, l)]).sum(0)
        assert torch.equal(torch.argmax(fused), torch.argmax(l[0]))


# ---------------------------------------------------------------- AVCD 결합식/EAD/plausibility

def make_avcd(faithful=True, alpha=2.5, tau=0.6):
    """setup() 없이 수식 필드만 구성한 AVCD 인스턴스."""
    m = AVCD()
    m.alpha = alpha
    m.tau = tau
    m.faithful = faithful
    if faithful:
        m.beta, m.fill = 0.2, -1e-4
    else:
        m.beta, m.fill = 0.1, -float("inf")
    return m


class TestAVCDFormula:
    def test_contrastive_coefficients(self):
        """결합식 (2+2α)·orig − 2α·out1 + out2 + out3 (공식 코드 L226)."""
        m = make_avcd(alpha=2.5)
        orig, o1, o2, o3 = (rand_logits(i + 20) for i in range(4))
        # plausibility가 개입 못 하게 cutoff를 전부 통과시키는 orig 사용
        orig_flat = torch.zeros(VOCAB)
        got = m.contrastive_step(orig_flat, o1, o2, o3)
        expect = (2 + 5.0) * orig_flat - 5.0 * o1 + o2 + o3
        # orig가 전부 0이면 cutoff = log(0.2) < 0 = orig → 마스킹 없음
        assert torch.allclose(got, expect, atol=1e-5)

    def test_plausibility_masks_low_prob_tokens(self):
        m = make_avcd(faithful=True, alpha=1.0)
        orig = torch.zeros(VOCAB)
        orig[0] = 10.0                       # 최고 logit
        o = torch.zeros(VOCAB)
        got = m.contrastive_step(orig, o, o, o)
        cutoff = math.log(m.beta) + 10.0     # ≈ 8.39
        below = orig < cutoff                # 토큰 0 제외 전부
        assert bool(below[1]) and not bool(below[0])
        assert torch.all(got[below] == m.fill)          # 코드 모드: -1e-4
        assert got[0] == pytest.approx((2 + 2) * 10.0)  # 통과 토큰은 결합식 값

    def test_plausibility_paper_mode_inf(self):
        m = make_avcd(faithful=False)
        orig = torch.zeros(VOCAB)
        orig[0] = 10.0
        got = m.contrastive_step(orig, torch.zeros(VOCAB), torch.zeros(VOCAB), torch.zeros(VOCAB))
        assert torch.isinf(got[1]) and got[1] < 0
        assert not torch.isinf(got[0])

    def test_entropy_and_ead_branch(self):
        m = make_avcd(tau=0.6)
        confident = torch.zeros(VOCAB); confident[3] = 50.0   # 엔트로피 ~0
        uniform = torch.zeros(VOCAB)                          # 엔트로피 = ln(100) ≈ 4.6
        assert m.entropy(confident) < 0.6 < m.entropy(uniform)


class TestAVCDMaskSpecs:
    def test_dominant_to_specs(self):
        from src.models import AVCD_MASK_SPECS
        assert AVCD_MASK_SPECS["language"] == ("VA", "A", "V")
        assert AVCD_MASK_SPECS["video"] == ("LA", "A", "L")
        assert AVCD_MASK_SPECS["audio"] == ("LV", "V", "L")
