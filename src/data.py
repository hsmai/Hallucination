"""벤치마크 공통 데이터 로더 (AVHBench / CMM).

공통 인터페이스: load_benchmark(cfg, "avhbench"|"cmm") -> list[Sample]

설계 규약 (blueprint D2, docs/data_report.md 참조):
- sample_id = "{video_id}::{question}" — 벤치마크 원본 식별자 기반, 자체 넘버링 금지.
  * AVHBench: video_id = 원본 'video_id' 필드 (예: "00159")
  * CMM:      video_id = 미디어 파일 stem (video 없으면 audio 기준, 예: "oxD5gs")
- 중복 (video_id, question) 키는 첫 등장만 유지 (MAD score.py의 dedup과 동일 규칙).
- 미디어 경로는 yaml의 prefix로 조립만 하고, skip_media_existence_check=true(로컬)면
  존재검사를 하지 않는다. 서버에서 QA json을 OURS 내부 파일로 교체할 때는
  yaml의 qa_json 경로만 바꾸면 된다.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, UNKNOWN

logger = logging.getLogger(__name__)

BENCHMARKS = ("avhbench", "cmm")

# AVHBench task명 (원본 json 표기 그대로)
AVH_TASK_VDAH = "Video-driven Audio Hallucination"
AVH_TASK_ADVH = "Audio-driven Video Hallucination"


@dataclass
class Sample:
    sample_id: str          # "{video_id}::{question}" — OURS와의 join 키
    video_id: str
    benchmark: str          # "AVHBench" | "CMM"
    category: str           # AVHBench: task / CMM: sub_category (지표 산출 단위)
    question: str           # 프롬프트 suffix가 붙지 않은 원본 질문
    ground_truth: str
    video_path: str | None  # prefix 조립 후 절대(또는 상대) 경로. 해당 modality 없으면 None
    audio_path: str | None
    extra: dict = field(default_factory=dict)  # CMM: category/modality/granularity 등


def _read_json(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"QA json이 없습니다: {path}\n"
            f"data/qa/ 확보 절차는 docs/data_report.md 참조."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"QA json 형식 오류(비어있거나 list가 아님): {path}")
    return data


def _join_prefix(prefix: str | None, rel: str | None) -> str | None:
    """미디어 경로 조립. prefix가 미확정(UNKNOWN)이면 상대경로 그대로 반환."""
    if rel is None:
        return None
    if not prefix or prefix == UNKNOWN:
        return rel
    return os.path.join(prefix, rel)


def _dedup(samples: list[Sample], benchmark: str) -> list[Sample]:
    """(sample_id) 중복 제거 — 첫 등장 유지 (MAD score.py와 동일 규칙)."""
    seen: set[str] = set()
    out: list[Sample] = []
    dropped = 0
    for s in samples:
        if s.sample_id in seen:
            dropped += 1
            continue
        seen.add(s.sample_id)
        out.append(s)
    if dropped:
        logger.warning("%s: 중복 sample_id %d건 제거(첫 등장 유지)", benchmark, dropped)
    return out


# ---------------------------------------------------------------- AVHBench

def load_avhbench(cfg: Config) -> list[Sample]:
    split = cfg.get("benchmarks.avhbench.split")
    media_dir = cfg.get("paths.avhbench_media_dir", None)

    if split == "full_qa_excluding_av_tasks":
        raw = _read_json(cfg.get("benchmarks.avhbench.qa_json"))
        raw = [x for x in raw if "AV" not in x["task"]]          # MAD 프로토콜
    elif split == "full_qa_all_tasks":
        raw = _read_json(cfg.get("benchmarks.avhbench.qa_json"))
    elif split == "avcd_val":
        raw = _read_json(cfg.get("benchmarks.avhbench.val_json", "data/qa/avhbench_val.json"))
    elif split == "avcd_test":
        raw = _read_json(cfg.get("benchmarks.avhbench.test_json", "data/qa/avhbench_test.json"))
    else:
        raise ValueError(
            f"benchmarks.avhbench.split 값이 잘못됨: {split!r} "
            f"(허용: full_qa_excluding_av_tasks | full_qa_all_tasks | avcd_val | avcd_test)"
        )

    # 서버 확정(2026-07-16): AVH_Bench에는 별도 audios/가 없고 mp4에 aac가 먹싱됨.
    # audio_path=None + extra.audio_in_video=True → 어댑터가 mp4에서 오디오를 얻는다
    # (Qwen: use_audio_in_video=True / VideoLLaMA2: va=True 또는 process_audio_from_video(mp4)).
    audio_in_video = bool(cfg.get("benchmarks.avhbench.audio_in_video", True))

    samples = []
    for x in raw:
        vid = str(x["video_id"])
        question = x.get("text") or x.get("question")
        gt = x.get("label") or x.get("answer")
        if question is None or gt is None:
            raise ValueError(f"AVHBench 항목 필드 누락: {x}")
        samples.append(Sample(
            sample_id=f"{vid}::{question}",
            video_id=vid,
            benchmark="AVHBench",
            category=x["task"],
            question=question,
            ground_truth=gt,
            video_path=_join_prefix(media_dir, f"videos/{vid}.mp4"),
            audio_path=None,
            extra={"task": x["task"], "split": split, "audio_in_video": audio_in_video},
        ))
    return _finalize(cfg, samples, "AVHBench")


# ---------------------------------------------------------------- CMM

def load_cmm(cfg: Config) -> list[Sample]:
    raw = _read_json(cfg.get("benchmarks.cmm.qa_json"))
    category_filter = cfg.get("benchmarks.cmm.categories")
    media_dir = cfg.get("paths.cmm_media_dir", None)

    if category_filter and category_filter != "all":
        raw = [x for x in raw if x.get("category") == category_filter]
        if not raw:
            raise ValueError(f"CMM category 필터 결과가 0건: {category_filter!r}")

    # 손상 비디오 blocklist (디코더 hang 유발) → _fixed/ 재인코딩본으로 교체 (OURS와 동일 처리)
    blocklist: set = set()
    bl_path = cfg.get("benchmarks.cmm.blocklist", None)
    if bl_path and Path(bl_path).exists():
        blocklist = {l.strip() for l in Path(bl_path).read_text().splitlines() if l.strip()}
    fixed_dir = cfg.get("benchmarks.cmm.fixed_dir", "_fixed")
    trap_emulation = bool(cfg.get("benchmarks.cmm.trap_emulation", False))
    if trap_emulation:
        logger.warning("CMM 함정 에뮬레이션 모드 — MAD repo식 입력(별도 wav 무시, 무음 mp4 트랙 사용). "
                       "검증 전용이며 본 결과에 사용 금지.")

    samples = []
    for x in raw:
        vpath = x.get("video_path")
        apath = x.get("audio_path")
        vpath = None if vpath in (None, "None", "") else vpath.lstrip("./")
        apath = None if apath in (None, "None", "") else apath.lstrip("./")
        anchor = vpath or apath
        if anchor is None:
            raise ValueError(f"CMM 항목에 video/audio 경로가 모두 없음: {x}")
        stem = os.path.splitext(os.path.basename(anchor))[0]
        if vpath and stem in blocklist:
            vpath = f"{fixed_dir}/{stem}.mp4"      # 재인코딩본으로 교체
        # 함정 에뮬레이션 (검증 전용, 기본 false): MAD repo의 CMM 입력 방식 재현 —
        # 별도 wav를 버리고 mp4의 (무음) 오디오 트랙을 audio branch에 공급.
        # VL2 VCD/MAD CMM 셀이 MAD 논문 수치와 다른 원인(무음 mp4 함정)의 실증 실험용.
        if trap_emulation and vpath:
            apath = None
        samples.append(Sample(
            sample_id=f"{stem}::{x['question']}",
            video_id=stem,
            benchmark="CMM",
            category=x["sub_category"],       # 지표 산출 단위 (V/A/L Dominance 등)
            question=x["question"],
            ground_truth=x["answer"],
            video_path=_join_prefix(media_dir, vpath),
            audio_path=_join_prefix(media_dir, apath),
            extra={
                "category_top": x.get("category"),
                "modality": x.get("modality"),
                "granularity": x.get("granularity"),
                "correlation_type": x.get("correlation_type"),
                # 함정 모드: mp4를 오디오 소스로 취급 (VL2 va=True / Qwen uaiv 경로)
                "audio_in_video": bool(trap_emulation and vpath),
            },
        ))
    return _finalize(cfg, samples, "CMM")


# ---------------------------------------------------------------- 공통

def _finalize(cfg: Config, samples: list[Sample], benchmark: str) -> list[Sample]:
    samples = _dedup(samples, benchmark)

    if not cfg.get("paths.skip_media_existence_check", True):
        missing = 0
        for s in samples:
            for p in (s.video_path, s.audio_path):
                if p and not os.path.isabs(p):
                    raise ValueError(
                        f"{benchmark}: 미디어 경로가 상대경로입니다({p}). "
                        f"paths.*_media_dir 프리픽스를 yaml에 설정하세요."
                    )
                if p and not os.path.exists(p):
                    missing += 1
                    logger.warning("%s: 미디어 없음 — %s (sample %s)", benchmark, p, s.sample_id)
        if missing:
            raise FileNotFoundError(
                f"{benchmark}: 미디어 파일 {missing}건 누락. "
                f"경로 프리픽스가 맞는지 확인하세요 (paths.avhbench_media_dir / paths.cmm_media_dir)."
            )

    logger.info("%s 로드 완료: %d샘플, 카테고리 분포 %s",
                benchmark, len(samples), dict(Counter(s.category for s in samples)))
    return samples


def load_benchmark(cfg: Config, name: str) -> list[Sample]:
    name = name.lower()
    if name == "avhbench":
        return load_avhbench(cfg)
    if name == "cmm":
        return load_cmm(cfg)
    raise ValueError(f"알 수 없는 벤치마크: {name!r} (허용: {BENCHMARKS})")


if __name__ == "__main__":
    # 빠른 점검용: python -m src.data
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .config import load_config

    cfg = load_config()
    for bench in BENCHMARKS:
        ss = load_benchmark(cfg, bench)
        cats = Counter(s.category for s in ss)
        print(f"\n[{bench}] {len(ss)} samples")
        for c, n in sorted(cats.items()):
            print(f"  {c}: {n}")
        print(f"  예시 sample_id: {ss[0].sample_id!r}")
        print(f"  예시 video_path: {ss[0].video_path!r}")
        print(f"  예시 audio_path: {ss[0].audio_path!r}")
