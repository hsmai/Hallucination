"""MAD — Modality-Adaptive Decoding (MAD 논문 Eq.9, repo 코드와 수식 일치 검증됨).

Step 1: head 프롬프트("Question: {q}\n" + modality query) 1회 forward
        → 'audio'/'video'/'both' 첫 토큰 logit softmax = (w_a, w_v, w_av). 질문당 1회 고정.
Step 2: branch [VA, V, A, T] 가중합 greedy:
        w = [2 + 2γ·w_av,  1 − γ(w_av − w_v),  1 − γ(w_av − w_a),  −γ(w_v + w_a)]
(MAD repo mm_contrast_decode L418-423 / 논문 Eq.9 전개 — docs/paper_settings.md §1)
"""

from __future__ import annotations

import time

import torch

from . import DecodingMethod


class MAD(DecodingMethod):
    name = "mad"

    def setup(self, adapter, cfg, benchmark):
        super().setup(adapter, cfg, benchmark)
        self.gamma = float(cfg.get("methods.mad.gamma"))
        self.query_prompt = cfg.get("prompts.mad_modality_query")
        self.question_wrap = cfg.get("prompts.mad_question_wrap")

    def head_prompt(self, sample) -> str:
        # MAD repo: "Question: " + (question+suffix) + "\n" + MODALITY_QUERY_PROMPT
        return self.question_wrap.format(question=self.question_with_suffix(sample)) + self.query_prompt

    @staticmethod
    def branch_weights(gamma: float, p_audio: float, p_video: float, p_both: float) -> list:
        """branch [VA, V, A, T] 가중치. tests/에서 Eq.9 전개와 대조 검증."""
        return [
            2 + 2 * gamma * p_both,
            1 - gamma * (p_both - p_video),
            1 - gamma * (p_both - p_audio),
            -gamma * (p_video + p_audio),
        ]

    def generate(self, sample) -> dict:
        t0 = time.time()
        ctx = self.adapter.prepare(sample, self.question_with_suffix(sample))

        p_audio, p_video, p_both = self.adapter.modality_query_probs(ctx, self.head_prompt(sample))
        weights = self.branch_weights(self.gamma, p_audio, p_video, p_both)

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
            # D2 internals 규약: w_v/w_a/w_av (blueprint) = (p_video, p_audio, p_both)
            "internals": {"mad": {"w_v": p_video, "w_a": p_audio, "w_av": p_both}},
            "inference_time_s": time.time() - t0,
        }
