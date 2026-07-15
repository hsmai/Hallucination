"""VCD-Extended — MAD 논문 Eq.10 (docs/paper_settings.md §1):

    logit = (1+3α)·logit_vaq − α·logit_ṽaq − α·logit_vãq − α·logit_ṽãq

˜ = modality 제거(ablation). branch 순서 [VA, V, A, T] 기준 가중치:
    [1+3α, −α, −α, −α]
(MAD repo mm_default_cd와 동일 계수. branch 매핑: ṽaq→A branch, vãq→V branch, ṽãq→T branch)
"""

from __future__ import annotations

import time

import torch

from . import DecodingMethod


class VCDExtended(DecodingMethod):
    name = "vcd_ext"

    def setup(self, adapter, cfg, benchmark):
        super().setup(adapter, cfg, benchmark)
        self.alpha = float(cfg.get("methods.vcd_ext.alpha"))
        if cfg.get("methods.vcd_ext.branch_mode") != "modality_ablation":
            raise NotImplementedError("branch_mode는 현재 modality_ablation만 지원")

    def generate(self, sample) -> dict:
        t0 = time.time()
        a = self.alpha
        # branch 순서 [VA, V, A, T]:
        #   VA=logit_vaq, V=logit_vãq(audio제거), A=logit_ṽaq(video제거), T=logit_ṽãq
        weights = [1 + 3 * a, -a, -a, -a]

        ctx = self.adapter.prepare(sample, self.question_with_suffix(sample))
        step_logits, state = self.adapter.branch_prefill(ctx)

        generated: list[int] = []
        for _ in range(self.max_new_tokens):
            fused = torch.stack([w * lg for lg, w in zip(step_logits, weights)]).sum(dim=0)
            nxt = int(torch.argmax(fused))
            generated.append(nxt)
            if nxt == self.adapter.eos_token_id:
                break
            step_logits = self.adapter.branch_step(state, nxt)

        return {
            "prediction": self.adapter.decode_tokens(generated),
            "internals": {"vcd_ext": {"alpha": a}},
            "inference_time_s": time.time() - t0,
        }
