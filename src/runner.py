"""공통 실험 러너.

사용:
  python -m src.runner --model videollama2_av --method mad --benchmark avhbench --dry-run
  python -m src.runner --model qwen2_5_omni_7b --method avcd --benchmark cmm [--limit 5]

동작:
- 모델 1회 로드(어댑터) → 샘플 루프 → 방법 플러그인 호출 → per-sample JSONL append
- 출력: results/{benchmark}/{model}__{method}.jsonl (+ 같은 이름 .meta.json 매니페스트)
- 중단-재개: 기존 JSONL의 sample_id를 읽어 처리분 skip (append 모드, 라인 단위 flush)
- D2 스키마(blueprint): sample_id, video_id, benchmark, category, question, ground_truth,
  method, model, prediction, correct(채점 단계에서 채움), internals, inference_time_s,
  seed, config_hash
- --dry-run: MockModel로 전 구간 검증 (로컬 전용, 실모델 로드 없음)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .data import load_benchmark
from .methods import METHOD_NAMES, get_method
from .models import load_adapter

logger = logging.getLogger("runner")


def out_paths(cfg, benchmark: str, model: str, method: str, dry_run: bool,
              out_tag: str = "") -> tuple[Path, Path]:
    root = Path(cfg.get("paths.results_dir", "results"))
    sub = "dryrun" if dry_run else "runs"
    d = root / sub / benchmark
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{method}" + (f"__{out_tag}" if out_tag else "")
    return d / f"{stem}.jsonl", d / f"{stem}.meta.json"


def load_done_ids(jsonl_path: Path) -> set:
    """기존 결과에서 처리 완료된 sample_id 수집 (깨진 마지막 라인은 무시)."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("깨진 결과 라인 무시: %.80s", line)
    return done


def git_commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def write_meta(meta_path: Path, args, cfg, n_total: int, n_done: int) -> None:
    """run manifest (blueprint D4). 재실행 시 히스토리 append."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "model": args.model, "method": args.method, "benchmark": args.benchmark,
        "dry_run": args.dry_run, "limit": args.limit,
        "config_hash": cfg.config_hash(),
        "config_pending": cfg.pending,
        "git_commit": git_commit_hash(),
        "n_total": n_total, "n_already_done": n_done,
    }
    history = []
    if meta_path.exists():
        try:
            history = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    history.append(entry)
    meta_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def apply_overrides(cfg, sets: list) -> None:
    """--set dotted.key=value 오버라이드 (게이트의 α 그리드/β 판정/경로 분리용).

    UNKNOWN_pending_server 노드는 _resolved_value를 교체하고, 일반 노드는 값을 교체한다.
    value는 JSON으로 파싱 시도(숫자/불리언), 실패 시 문자열.
    """
    from .config import UNKNOWN
    for item in sets:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--set 형식 오류: {item!r} (dotted.key=value)")
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        node = cfg._resolved
        parts = key.split(".")
        for p in parts[:-1]:
            if isinstance(node, dict) and node.get("status") == UNKNOWN:
                node = node["_resolved_value"]
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"--set 경로 없음: {key} (막힌 지점: {p})")
            node = node[p]
        if isinstance(node, dict) and node.get("status") == UNKNOWN:
            node = node["_resolved_value"]
        leaf = parts[-1]
        if leaf not in node:
            raise KeyError(f"--set 경로 없음: {key} (막힌 지점: {leaf})")
        if isinstance(node[leaf], dict) and node[leaf].get("status") == UNKNOWN:
            node[leaf]["_resolved_value"] = parsed
        else:
            node[leaf] = parsed
        logger.info("override: %s = %r", key, parsed)


def run(args) -> int:
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    seed = cfg.get("experiment.seed")
    config_hash = cfg.config_hash()

    samples = load_benchmark(cfg, args.benchmark)
    if args.ids_file:
        wanted = {l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()}
        samples = [s for s in samples if s.sample_id in wanted]
        missing = wanted - {s.sample_id for s in samples}
        if missing:
            logger.warning("ids-file 중 %d개는 벤치마크에 없음 (예: %s)",
                           len(missing), next(iter(missing)))
    if args.limit:
        samples = samples[: args.limit]

    if args.max_new_tokens:  # 정성 샘플 재생성용 오버라이드 (D3)
        node = cfg._resolved["decoding"]["max_new_tokens"][args.benchmark]
        if isinstance(node, dict):
            node["_resolved_value"] = args.max_new_tokens
        else:
            cfg._resolved["decoding"]["max_new_tokens"][args.benchmark] = args.max_new_tokens

    adapter = load_adapter(args.model, cfg, dry_run=args.dry_run, method=args.method)
    method = get_method(args.method)
    method.setup(adapter, cfg, args.benchmark)

    jsonl_path, meta_path = out_paths(cfg, args.benchmark, args.model, args.method,
                                      args.dry_run, args.out_tag)
    done = load_done_ids(jsonl_path)
    todo = [s for s in samples if s.sample_id not in done]
    logger.info("[%s×%s×%s] 전체 %d, 완료 %d, 남음 %d → %s",
                args.model, args.method, args.benchmark,
                len(samples), len(samples) - len(todo), len(todo), jsonl_path)
    write_meta(meta_path, args, cfg, n_total=len(samples), n_done=len(samples) - len(todo))

    if not todo:
        logger.info("처리할 샘플 없음 (모두 완료).")
        return 0

    n_err = 0
    t_start = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(todo):
            try:
                out = method.generate(sample)
                record = {
                    "sample_id": sample.sample_id,
                    "video_id": sample.video_id,
                    "benchmark": sample.benchmark,
                    "category": sample.category,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                    "method": args.method,
                    "model": args.model,
                    "prediction": out["prediction"],
                    "correct": None,  # src/score.py가 채움
                    "internals": out.get("internals", {}),
                    "inference_time_s": round(out.get("inference_time_s", 0.0), 4),
                    "seed": seed,
                    "config_hash": config_hash,
                }
            except KeyboardInterrupt:
                logger.warning("중단됨 — 재실행 시 %s부터 이어서 처리", sample.sample_id)
                raise
            except Exception as e:
                n_err += 1
                logger.error("샘플 실패 %s: %s", sample.sample_id, e)
                record = {
                    "sample_id": sample.sample_id, "video_id": sample.video_id,
                    "benchmark": sample.benchmark, "category": sample.category,
                    "question": sample.question, "ground_truth": sample.ground_truth,
                    "method": args.method, "model": args.model,
                    "prediction": f"ERROR: {e}", "correct": None, "internals": {},
                    "inference_time_s": 0.0, "seed": seed, "config_hash": config_hash,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % args.log_interval == 0:
                rate = (i + 1) / (time.time() - t_start)
                logger.info("  %d/%d (%.1f samples/s, err %d)", i + 1, len(todo), rate, n_err)

    logger.info("완료: %d처리 / 에러 %d / %.1fs", len(todo), n_err, time.time() - t_start)
    return 0 if n_err == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="비교군 실험 러너")
    p.add_argument("--model", required=True, help="yaml models.* 키")
    p.add_argument("--method", required=True, choices=METHOD_NAMES)
    p.add_argument("--benchmark", required=True, choices=("avhbench", "cmm"))
    p.add_argument("--config", default=None, help="기본: configs/unified_settings.yaml")
    p.add_argument("--dry-run", action="store_true", help="MockModel로 배관 검증 (로컬)")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (스모크용)")
    p.add_argument("--ids-file", default=None, help="sample_id 목록 파일 — 해당 샘플만 실행 (D3 재생성)")
    p.add_argument("--max-new-tokens", type=int, default=0, help="디코딩 길이 오버라이드 (D3 서술형 재생성)")
    p.add_argument("--out-tag", default="", help="출력 파일 구분 태그 — 재생성 실행 시 본 결과와 분리 (예: regen256)")
    p.add_argument("--set", action="append", default=[],
                   help="설정 오버라이드 dotted.key=value (반복 가능) — 게이트 α그리드/β판정/경로분리용")
    p.add_argument("--log-interval", type=int, default=200)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(run(build_parser().parse_args()))
