"""실모델 어댑터 레지스트리 (서버 전용).

로컬(dry-run)에서는 이 패키지의 common.py만 사용된다 — 실모델 어댑터 모듈은
transformers/videollama2 import가 필요하므로 지연 로드한다.
"""

from __future__ import annotations


def _build_videollama2(cfg, method: str):
    from .videollama2_av import VideoLLaMA2Adapter
    return VideoLLaMA2Adapter(cfg, method)


def _build_qwen(cfg, method: str):
    from .qwen_omni import QwenOmniAdapter
    return QwenOmniAdapter(cfg, method)


REAL_ADAPTERS = {
    "videollama2_av": _build_videollama2,
    "qwen2_5_omni_7b": _build_qwen,
}
