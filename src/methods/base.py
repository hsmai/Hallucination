"""Base: 모델 원본 greedy 디코딩 (temperature=0, do_sample=False)."""

from __future__ import annotations

import time

from . import DecodingMethod


class BaseDecoding(DecodingMethod):
    name = "base"

    def generate(self, sample) -> dict:
        t0 = time.time()
        ctx = self.adapter.prepare(sample, self.question_with_suffix(sample))
        prediction = self.adapter.greedy_generate(ctx, self.max_new_tokens)
        return {
            "prediction": prediction,
            "internals": {},
            "inference_time_s": time.time() - t0,
        }
