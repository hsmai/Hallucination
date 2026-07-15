"""채점기 — MAD repo score.py 로직 이식 (임시 채점기).

이식 원본: third_party/MAD/VideoLLaMA2/score.py (qwen-omni/score.py와 동일 파일)
- extract_answer: yes 패턴(yes|true|correct|affirmative)을 no 패턴보다 먼저 검사
- normalize_answer: lowercase/strip, 끝 문장부호 제거, 동의어 매핑, substring yes/no
- ERROR: 예측 제외, (video_id, question) 중복은 로더에서 이미 제거됨

교체 규약 (blueprint: 서버에서 OURS 채점기로 교체):
- is_correct(prediction, ground_truth)가 유일한 판정 진입점.
  OURS 채점기 확보 시 이 함수만 갈아끼우면 된다 (yaml scoring.scorer로 선택).

CLI:  python -m src.score --jsonl results/dryrun/avhbench/*.jsonl
      → 각 파일의 correct 필드를 채워 rewrite + 카테고리별 정확도 출력
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- MAD 로직 (원본 그대로)

YES_PATTERNS = [r"\byes\b", r"\btrue\b", r"\bcorrect\b", r"\baffirmative\b"]
NO_PATTERNS = [r"\bno\b", r"\bfalse\b", r"\bincorrect\b", r"\bnegative\b"]


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]$", "", answer)
    if answer in ["yes", "y", "true", "1", "correct"]:
        return "yes"
    elif answer in ["no", "n", "false", "0", "incorrect"]:
        return "no"
    if "yes" in answer:
        return "yes"
    elif "no" in answer:
        return "no"
    return answer


def extract_answer(prediction: str) -> str:
    prediction = prediction.strip()
    prediction_lower = prediction.lower()
    for pattern in YES_PATTERNS:          # yes 패턴 우선 (원본 순서 유지)
        if re.search(pattern, prediction_lower):
            return "yes"
    for pattern in NO_PATTERNS:
        if re.search(pattern, prediction_lower):
            return "no"
    words = prediction.split()
    if words:
        return normalize_answer(words[0])
    return prediction.lower().strip()


def mad_is_correct(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(extract_answer(prediction)) == normalize_answer(ground_truth)


# ---------------------------------------------------------------- 채점기 선택

SCORERS = {
    "mad_score": mad_is_correct,
    # "ours": 서버에서 omni-steering 채점 로직 이식 후 등록 (yaml scoring.scorer 변경)
}


def get_scorer(cfg):
    name = cfg.get("scoring.scorer")
    if name not in SCORERS:
        raise ValueError(f"알 수 없는 채점기: {name!r} (등록: {sorted(SCORERS)})")
    return SCORERS[name]


# ---------------------------------------------------------------- JSONL 채점

def score_jsonl(path: Path, is_correct) -> dict:
    """correct 필드를 채워 파일을 rewrite하고 카테고리별 집계 반환. 재실행 안전."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    n_err = 0
    for r in records:
        if r["prediction"].startswith("ERROR:"):
            r["correct"] = None
            n_err += 1
        else:
            r["correct"] = bool(is_correct(r["prediction"], r["ground_truth"]))

    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)

    by_cat = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    for r in records:
        if r["correct"] is None:
            continue
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["correct"])
    total_c = sum(c for c, _ in by_cat.values())
    total_n = sum(n for _, n in by_cat.values())
    return {
        "file": str(path), "n": len(records), "n_error": n_err,
        "overall": {"correct": total_c, "total": total_n,
                    "acc": total_c / total_n if total_n else 0.0},
        "by_category": {k: {"correct": c, "total": n, "acc": c / n if n else 0.0}
                        for k, (c, n) in sorted(by_cat.items())},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="JSONL 채점 (correct 필드 채움)")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(args.config)
    is_correct = get_scorer(cfg)

    for pattern in args.jsonl:
        paths = sorted(Path(".").glob(pattern)) if any(ch in pattern for ch in "*?[") \
            else [Path(pattern)]
        if not paths:
            print(f"경고: 매칭 파일 없음 — {pattern}", file=sys.stderr)
        for p in paths:
            stats = score_jsonl(p, is_correct)
            print(f"\n{p}  (n={stats['n']}, error={stats['n_error']})")
            print(f"  Overall: {stats['overall']['acc']*100:.2f}% "
                  f"({stats['overall']['correct']}/{stats['overall']['total']})")
            for cat, s in stats["by_category"].items():
                print(f"  {cat}: {s['acc']*100:.2f}% ({s['correct']}/{s['total']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
