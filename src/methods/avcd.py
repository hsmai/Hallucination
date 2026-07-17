"""AVCD — Audio-Visual Contrastive Decoding (NeurIPS 2025, 공식 코드 대조 검증됨).

스텝별 절차 (AVCD/videollama2/__init__.py generate_long 경로, docs/code_analysis.md §2.5):
  1. 원본 forward → logits, dominance 순위, masking threshold
  2. EAD: entropy(softmax(logits)) < τ 이면 CD skip (원본 logits 사용)
  3. 아니면 dominant에 따라 mask_spec 3종 forward 후 결합:
       contrast = (2+2α)·orig − 2α·out1 + out2 + out3
  4. adaptive plausibility: cutoff = log(β) + max(orig);
       faithful_mode(코드 재현): contrast[orig < cutoff] = -1e-4, β=0.2
       paper mode:               contrast[orig < cutoff] = -inf,  β=0.1
  5. greedy argmax

α는 벤치마크별 (yaml methods.avcd.alpha.{avhbench,cmm}).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from . import DecodingMethod
from ..models import AVCD_MASK_SPECS


class AVCD(DecodingMethod):
    name = "avcd"

    def setup(self, adapter, cfg, benchmark):
        super().setup(adapter, cfg, benchmark)
        alpha = cfg.get(f"methods.avcd.alpha.{benchmark}")
        if isinstance(alpha, dict):               # 모델별 확정값 (게이트 그리드 결과)
            alpha = alpha[getattr(adapter, "model_key", None)]
        self.alpha = float(alpha)
        self.tau = float(cfg.get("methods.avcd.ead_tau"))
        faithful = bool(cfg.get("methods.avcd.faithful_mode"))
        if faithful:
            self.beta = float(cfg.get("methods.avcd.plausibility_beta"))
            self.fill = float(cfg.get("methods.avcd.implausible_fill"))
        else:
            self.beta = float(cfg.get("methods.avcd.plausibility_beta_paper"))
            self.fill = -float("inf")
        self.faithful = faithful

    @staticmethod
    def entropy(logits: torch.Tensor) -> float:
        p = F.softmax(logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        return float(-(p * logp).sum())

    def contrastive_step(self, orig: torch.Tensor, out1: torch.Tensor,
                         out2: torch.Tensor, out3: torch.Tensor) -> torch.Tensor:
        """결합식 + plausibility. tests/에서 계수·마스킹 검증."""
        a = self.alpha
        contrast = (2 + 2 * a) * orig - 2 * a * out1 + out2 + out3
        cutoff = math.log(self.beta) + float(orig.max())
        return contrast.masked_fill(orig < cutoff, self.fill)

    def generate(self, sample) -> dict:
        t0 = time.time()
        ctx = self.adapter.prepare(sample, self.question_with_suffix(sample))

        generated: list[int] = []
        first_dominant = None
        skipped = 0
        steps = 0

        for _ in range(self.max_new_tokens):
            orig, dominance, threshold = self.adapter.avcd_orig_forward(ctx, generated)
            steps += 1
            if first_dominant is None:
                first_dominant = dominance[0][0]

            if self.entropy(orig) < self.tau:
                next_logits = orig          # EAD: 저엔트로피 → CD skip
                skipped += 1
            else:
                m1, m2, m3 = AVCD_MASK_SPECS[dominance[0][0]]
                out1 = self.adapter.avcd_masked_forward(ctx, generated, m1, threshold)
                out2 = self.adapter.avcd_masked_forward(ctx, generated, m2, threshold)
                out3 = self.adapter.avcd_masked_forward(ctx, generated, m3, threshold)
                next_logits = self.contrastive_step(orig, out1, out2, out3)

            nxt = int(torch.argmax(next_logits))
            generated.append(nxt)
            if nxt == self.adapter.eos_token_id:
                break

        return {
            "prediction": self.adapter.decode_tokens(generated),
            "internals": {"avcd": {
                "dominant": first_dominant,
                "ead_skipped_ratio": skipped / steps if steps else 0.0,
                "alpha": self.alpha,
                "faithful_mode": self.faithful,
            }},
            "inference_time_s": time.time() - t0,
        }
