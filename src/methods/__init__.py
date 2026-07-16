"""방법 플러그인 레지스트리.

DecodingMethod 인터페이스 (claude_code_master_prompt.md Phase L3):
    class DecodingMethod:
        def setup(self, model, cfg): ...
        def generate(self, sample) -> dict   # {"prediction": str, "internals": {...}}
"""

from __future__ import annotations


class DecodingMethod:
    name: str = "abstract"

    def setup(self, adapter, cfg, benchmark: str) -> None:
        """adapter: src.models.ModelAdapter, cfg: src.config.Config, benchmark: avhbench|cmm"""
        self.adapter = adapter
        self.cfg = cfg
        self.benchmark = benchmark
        self.max_new_tokens = cfg.get(f"decoding.max_new_tokens.{benchmark}")
        # 프롬프트 suffix는 모델별 (2026-07-16 OURS 대조 확정 — 각 모델 자체 eval 프롬프트)
        suffix_key = "prompts.avhbench_suffix" if benchmark == "avhbench" else "prompts.cmm_suffix"
        suffix = cfg.get(suffix_key)
        if isinstance(suffix, dict):
            model_key = getattr(adapter, "model_key", None)
            if model_key not in suffix:
                raise KeyError(f"{suffix_key}에 모델 키 {model_key!r} 없음 (yaml 확인)")
            suffix = suffix[model_key]
        self.prompt_suffix = suffix

    def question_with_suffix(self, sample) -> str:
        return sample.question + self.prompt_suffix

    def generate(self, sample) -> dict:
        raise NotImplementedError


def get_method(name: str) -> DecodingMethod:
    from .base import BaseDecoding
    from .vcd_ext import VCDExtended
    from .mad import MAD
    from .avcd import AVCD

    registry = {
        "base": BaseDecoding,
        "vcd_ext": VCDExtended,
        "mad": MAD,
        "avcd": AVCD,
    }
    if name not in registry:
        raise ValueError(f"알 수 없는 방법: {name!r} (허용: {sorted(registry)})")
    return registry[name]()


METHOD_NAMES = ("base", "vcd_ext", "mad", "avcd")
