"""모델 어댑터 인터페이스 + MockModel (dry-run용).

방법 플러그인(src/methods/)은 이 인터페이스에만 의존한다. 실제 모델 어댑터
(VideoLLaMA2-AV, Qwen2.5-Omni)는 L4/서버 단계에서 이 인터페이스를 구현한다
(제약 1: 로컬에서 7B 모델 로드 금지 — 로컬에는 MockAdapter만 존재).

인터페이스가 요구하는 연산은 docs/code_analysis.md 의 분석에 근거한다:
- 4-branch CD (MAD/VCD-ext): branch = ("va","v","a","t") modality-ablation 프롬프트,
  branch별 KV 캐시 유지, 매 스텝 동일 토큰 공급 (MAD repo mm_contrast_decode 구조)
- MAD weight 질의: head 프롬프트 1회 forward → 'audio'/'video'/'both' 첫 토큰 logit
- AVCD: 원본 forward에서 (logits, dominance 순위, masking threshold) 산출,
  마스킹 forward는 mask_spec("V","A","L","VA","LA","LV")별 logits 반환
  (공식 코드 generate_long 경로와 동일하게 스텝마다 전체 시퀀스 re-forward)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch

from .data import Sample

logger = logging.getLogger(__name__)

BRANCHES = ("va", "v", "a", "t")  # 순서 고정: [VA, V만, A만, 텍스트만]

# AVCD mask_spec: dominant modality에 따라 덜 지배적인 modality 조합을 마스킹
# (AVCD/videollama2/__init__.py L178-189 그대로)
AVCD_MASK_SPECS = {
    "language": ("VA", "A", "V"),
    "video": ("LA", "A", "L"),
    "audio": ("LV", "V", "L"),
}


class ModelAdapter:
    """방법 플러그인이 사용하는 모델 연산 인터페이스."""

    name: str = "abstract"
    eos_token_id: int = -1

    # ---- 공통 ----
    def prepare(self, sample: Sample, question_with_suffix: str) -> Any:
        """샘플별 컨텍스트(미디어 텐서/프롬프트) 준비. 반환값은 이 어댑터 전용 핸들."""
        raise NotImplementedError

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    # ---- base ----
    def greedy_generate(self, ctx: Any, max_new_tokens: int) -> str:
        raise NotImplementedError

    # ---- 4-branch CD (MAD / VCD-ext) ----
    def branch_prefill(self, ctx: Any) -> Tuple[List[torch.Tensor], Any]:
        """4개 branch 프롬프트를 각각 prefill. (branch별 마지막 logits[vocab], 상태) 반환.
        audio가 없는 샘플(CMM 일부)은 'a' branch를 텍스트만으로 대체한다(어댑터 책임)."""
        raise NotImplementedError

    def branch_step(self, state: Any, token_id: int) -> List[torch.Tensor]:
        """모든 branch에 동일 토큰 공급 후 branch별 다음 logits 반환."""
        raise NotImplementedError

    # ---- MAD ----
    def modality_query_probs(self, ctx: Any, query_prompt: str) -> Tuple[float, float, float]:
        """head 프롬프트 forward → (p_audio, p_video, p_both) softmax 확률."""
        raise NotImplementedError

    # ---- AVCD ----
    def avcd_orig_forward(self, ctx: Any, generated_ids: Sequence[int]) -> Tuple[torch.Tensor, list, float]:
        """마스킹 없는 forward. (마지막 logits, dominance 내림차순 [(mod, score)...], threshold)."""
        raise NotImplementedError

    def avcd_masked_forward(self, ctx: Any, generated_ids: Sequence[int],
                            mask_spec: str, threshold: float) -> torch.Tensor:
        """mask_spec에 해당하는 modality span의 고-attention 토큰을 zeroing한 forward."""
        raise NotImplementedError


# ---------------------------------------------------------------- MockModel

class MockAdapter(ModelAdapter):
    """dry-run용 가짜 모델.

    - vocab 16: id 0=EOS, 1="Yes", 2="No", 3..15 더미
    - 모든 logits는 (seed, sample_id, 용도, step) 해시로 결정적 생성
      → 중단-재개 시 동일 출력 보장, 실행 간 재현 가능
    - 실제 수식 경로(가중합/EAD/plausibility)를 그대로 태우는 것이 목적
    """

    VOCAB = 16
    EOS, YES, NO = 0, 1, 2

    def __init__(self, model_key: str, seed: int = 42):
        self.name = f"mock:{model_key}"
        self.model_key = model_key
        self.seed = seed
        self.eos_token_id = self.EOS

    # 결정적 RNG
    def _rng(self, *parts: Any) -> torch.Generator:
        key = "|".join(str(p) for p in (self.seed, self.model_key, *parts))
        h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        g = torch.Generator()
        g.manual_seed(h)
        return g

    def _logits(self, *parts: Any) -> torch.Tensor:
        g = self._rng(*parts)
        logits = torch.randn(self.VOCAB, generator=g)
        # Yes/No가 대체로 이기게 + 2스텝쯤에 EOS가 나오게 바이어스
        logits[self.YES] += 2.0
        logits[self.NO] += 2.0
        step = parts[-1] if isinstance(parts[-1], int) else 0
        logits[self.EOS] += 4.0 * step  # step 1부터 EOS 우세
        return logits

    def prepare(self, sample: Sample, question_with_suffix: str) -> dict:
        return {
            "sample_id": sample.sample_id,
            "question": question_with_suffix,
            "has_audio": sample.audio_path is not None or bool(sample.extra.get("audio_in_video")),
            "has_video": sample.video_path is not None,
        }

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        words = {self.YES: "Yes", self.NO: "No"}
        toks = [words.get(t, f"<tok{t}>") for t in token_ids if t != self.EOS]
        return " ".join(toks) if toks else ""

    def greedy_generate(self, ctx: dict, max_new_tokens: int) -> str:
        out = []
        for step in range(max_new_tokens):
            logits = self._logits(ctx["sample_id"], "base", step)
            nxt = int(torch.argmax(logits))
            if nxt == self.EOS:
                break
            out.append(nxt)
        return self.decode_tokens(out)

    def branch_prefill(self, ctx: dict) -> Tuple[List[torch.Tensor], dict]:
        state = {"sample_id": ctx["sample_id"], "step": 0}
        logits = [self._logits(ctx["sample_id"], f"branch:{b}", 0) for b in BRANCHES]
        return logits, state

    def branch_step(self, state: dict, token_id: int) -> List[torch.Tensor]:
        state["step"] += 1
        return [self._logits(state["sample_id"], f"branch:{b}", token_id, state["step"])
                for b in BRANCHES]

    def modality_query_probs(self, ctx: dict, query_prompt: str) -> Tuple[float, float, float]:
        g = self._rng(ctx["sample_id"], "modality_query", query_prompt[:16])
        z = torch.randn(3, generator=g)
        p = torch.softmax(z, dim=0).tolist()
        return p[0], p[1], p[2]  # (p_audio, p_video, p_both)

    def avcd_orig_forward(self, ctx: dict, generated_ids: Sequence[int]):
        step = len(generated_ids)
        logits = self._logits(ctx["sample_id"], "avcd:orig", step)
        g = self._rng(ctx["sample_id"], "avcd:dom")
        scores = torch.rand(3, generator=g).tolist()
        dominance = sorted(zip(("language", "video", "audio"), scores),
                           key=lambda x: x[1], reverse=True)
        threshold = 0.5
        return logits, dominance, threshold

    def avcd_masked_forward(self, ctx: dict, generated_ids: Sequence[int],
                            mask_spec: str, threshold: float) -> torch.Tensor:
        step = len(generated_ids)
        return self._logits(ctx["sample_id"], f"avcd:mask:{mask_spec}", step)


# ---------------------------------------------------------------- 로더

def load_adapter(model_key: str, cfg, dry_run: bool, method: str = "base") -> ModelAdapter:
    """model_key: unified_settings.yaml models.* 키 (videollama2_av | qwen2_5_omni_7b).

    method가 필요한 이유: VideoLLaMA2×AVCD는 공식 fork 백엔드(third_party/AVCD)를,
    그 외는 vanilla(third_party/VideoLLaMA2)를 로드한다 (fork들이 패키지명을 공유하므로
    run당 1프로세스에서 하나만 import 가능 — src/adapters/videollama2_av.py 참조).
    """
    if model_key not in cfg.get("models"):
        raise ValueError(f"알 수 없는 모델 키: {model_key!r} (yaml models.* 참조)")
    if dry_run:
        return MockAdapter(model_key, seed=cfg.get("experiment.seed"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "실모델 실행은 GPU 서버 전용입니다 (CLAUDE.md 제약 1). "
            "로컬에서는 --dry-run을 사용하세요.")
    from .adapters import REAL_ADAPTERS
    return REAL_ADAPTERS[model_key](cfg, method)
