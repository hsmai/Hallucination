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
import dataclasses
import datetime
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config
from .data import AVH_TASK_ADVH, AVH_TASK_VDAH, load_benchmark
from .methods import METHOD_NAMES, get_method
from .models import load_adapter
from .score import mad_is_correct, extract_answer, normalize_answer

logger = logging.getLogger("runner")


# ---------------------------------------------------------------- 모니터링 (체크포인트)

class RunMonitor:
    """실행 중 상태 확인 (2026-07-15 사용자 지시).

    - 매 log-interval: 한 줄 상태 (누적acc, 예측분포, internals 요약)
    - 전체의 1/4 지점마다 체크포인트 블록:
      [정량] 카테고리별 누적 acc vs gate_targets Δ (잠정치 — 임시 채점기·부분 샘플)
      [정성] 해당 시점 샘플의 본실험 결과 + 서술 프로브(공식 디코딩 × MAD 그림 프롬프트)
             + GT 캡션 + 프레임 저장 (모니터링 전용 — 본 실험 JSONL 불변)
    모니터링 실패는 본 실험을 중단시키지 않는다 (전부 방어적).
    """

    # 카테고리 → (표시명, gate_targets 배열 인덱스)
    CMM_ORDER = [("overrely_visual_ignore_audio", "Visual Dom.", 0),
                 ("overrely_audio_ignore_visual", "Audio Dom.", 1),
                 ("overrely_language_ignore_visual", "Language Dom.", 2)]
    AVH_ORDER = [(AVH_TASK_VDAH, "Video-Driven Audio Hall.", 0),
                 (AVH_TASK_ADVH, "Audio-Driven Video Hall.", 1)]

    def __init__(self, cfg, args, n_todo: int, jsonl_path: Path):
        self.enabled = bool(cfg.get("monitoring.enabled", True)) and n_todo > 0
        self.cfg = cfg
        self.benchmark = args.benchmark
        self.probe_prompt = cfg.get("monitoring.probe_prompt", "")
        self.probe_max_new_tokens = int(cfg.get("monitoring.probe_max_new_tokens", 256))
        self.probe_frames = int(cfg.get("monitoring.probe_frames", 2))
        fracs = cfg.get("monitoring.checkpoints", [0.25, 0.5, 0.75, 1.0])
        self.marks = sorted({min(n_todo, max(1, math.ceil(n_todo * f))) for f in fracs})
        self.ckpt_dir = jsonl_path.parent / "checkpoints" / jsonl_path.stem
        self.targets = self._load_targets(args.model, args.method)
        self.cat_tally = defaultdict(lambda: [0, 0])   # category -> [correct, total]
        self.pred_dist = Counter()
        self.mad_w = defaultdict(float)
        self.avcd_dom = Counter()
        self.avcd_skip_sum = 0.0
        self.n_method_records = 0
        self.n_ckpt = 0
        self._captions = None

    def _load_targets(self, model: str, method: str):
        try:
            t = self.cfg.get("gate_targets.mad_paper_table1")[model][method]
            vals = t["cmm" if self.benchmark == "cmm" else "avh"]
            order = self.CMM_ORDER if self.benchmark == "cmm" else self.AVH_ORDER
            out = {cat: vals[idx] for cat, _, idx in order}
            out["__overall__"] = vals[-1]
            return out
        except (KeyError, TypeError, IndexError):
            return {}

    def _caption(self, video_id: str) -> str:
        if self.benchmark != "avhbench":
            return ""
        if self._captions is None:
            try:
                qa = json.loads(Path(self.cfg.get("benchmarks.avhbench.qa_json")).read_text())
                self._captions = {x["video_id"]: x["label"] for x in qa
                                  if x.get("task") == "AV Captioning"}
            except Exception:
                self._captions = {}
        return self._captions.get(video_id, "")

    # ---- 집계 갱신 ----
    def observe(self, record: dict) -> None:
        if not self.enabled:
            return
        pred = record["prediction"]
        if pred.startswith("ERROR:"):
            return
        norm = normalize_answer(extract_answer(pred))
        self.pred_dist[norm if norm in ("yes", "no") else "other"] += 1
        c = self.cat_tally[record["category"]]
        c[1] += 1
        c[0] += int(mad_is_correct(pred, record["ground_truth"]))
        internals = record.get("internals", {})
        if "mad" in internals:
            for k in ("w_v", "w_a", "w_av"):
                self.mad_w[k] += internals["mad"].get(k, 0.0)
            self.n_method_records += 1
        elif "avcd" in internals:
            self.avcd_dom[internals["avcd"].get("dominant")] += 1
            self.avcd_skip_sum += internals["avcd"].get("ead_skipped_ratio", 0.0)
            self.n_method_records += 1

    def seed_from_records(self, records: list) -> None:
        """중단-재개 시 기존 결과로 누적치 초기화 (누적 지표가 전체 기준이 되도록)."""
        for r in records:
            self.observe(r)

    # ---- 출력 ----
    def _overall(self):
        c = sum(v[0] for v in self.cat_tally.values())
        n = sum(v[1] for v in self.cat_tally.values())
        return (c / n * 100 if n else 0.0), n

    def oneline_suffix(self) -> str:
        if not self.enabled:
            return ""
        acc, n = self._overall()
        total = sum(self.pred_dist.values()) or 1
        dist = "/".join(f"{k} {v * 100 // total}%" for k, v in self.pred_dist.most_common(3))
        extra = ""
        if self.n_method_records and self.mad_w:
            m = {k: v / self.n_method_records for k, v in self.mad_w.items()}
            extra = f" | mad w̄_v {m['w_v']:.2f} w̄_a {m['w_a']:.2f} w̄_av {m['w_av']:.2f}"
        elif self.n_method_records and (self.avcd_dom or self.avcd_skip_sum):
            top = self.avcd_dom.most_common(1)
            dom = f"{top[0][0]} {top[0][1] * 100 // self.n_method_records}%" if top else "-"
            extra = f" | avcd dom {dom}, skip {self.avcd_skip_sum / self.n_method_records * 100:.0f}%"
        return f" | 누적acc {acc:.1f}% (n={n}) | pred {dist}{extra}"

    def maybe_checkpoint(self, i_done: int, sample, record: dict, method) -> None:
        if not self.enabled or i_done not in self.marks:
            return
        self.n_ckpt += 1
        try:
            self._emit_checkpoint(i_done, sample, record, method)
        except Exception as e:  # 모니터링은 본 실험을 절대 방해하지 않는다
            logger.warning("체크포인트 %d 출력 실패(본 실험 계속): %s", self.n_ckpt, e)

    def _emit_checkpoint(self, i_done: int, sample, record: dict, method) -> None:
        k, total_k = self.n_ckpt, len(self.marks)
        lines = ["", "═" * 70,
                 f"══ 체크포인트 {k}/{total_k} ({i_done}번째 처리) — {record['model']} × "
                 f"{record['method']} × {self.benchmark} ══"]

        # [정량] 누적 vs 목표
        lines.append("[정량] 누적 지표 vs 목표 (잠정치: 임시 채점기·부분 샘플 — 순서 편향 주의)")
        order = self.CMM_ORDER if self.benchmark == "cmm" else self.AVH_ORDER
        for cat, col, _ in order:
            c, n = self.cat_tally.get(cat, [0, 0])
            if n == 0:
                lines.append(f"  {col:<26}: (아직 없음)")
                continue
            acc = c / n * 100
            t = self.targets.get(cat)
            tail = f"(목표 {t:.1f} | Δ {acc - t:+.1f})" if t is not None else "(목표 없음)"
            lines.append(f"  {col:<26}: {acc:5.1f}%  {tail}  [n={n}]")
        acc, n = self._overall()
        t = self.targets.get("__overall__")
        tail = f"(목표 {t:.1f} | Δ {acc - t:+.1f})" if t is not None else ""
        lines.append(f"  {'Overall':<26}: {acc:5.1f}%  {tail}  [n={n}]")

        # [정성] 본실험 결과 + 캡션
        ok = record["correct"] if record["correct"] is not None else \
            mad_is_correct(record["prediction"], record["ground_truth"])
        lines += ["[정성] 샘플 점검 (이 지점에서 처리된 샘플)",
                  f"  sample_id : {record['sample_id']}",
                  f"  video     : {sample.video_path or '-'}   audio: {sample.audio_path or '-'}"]
        cap = self._caption(record["video_id"])
        if cap:
            lines.append(f"  장면설명(GT캡션): {cap}")
        elif sample.extra:
            lines.append(f"  메타      : {json.dumps(sample.extra, ensure_ascii=False)}")
        lines += [f"  본실험 Q  : {record['question']}",
                  f"  GT: {record['ground_truth']} | 예측(raw): {record['prediction']!r} "
                  f"[{'정답' if ok else '오답'}]"]
        if record.get("internals"):
            lines.append(f"  internals : {json.dumps(record['internals'], ensure_ascii=False)}")

        # [정성] 서술 프로브 — 공식 디코딩 그대로, 프롬프트만 MAD Fig.9-10 문구
        probe_out = None
        if self.probe_prompt:
            probe_out = self._run_probe(method, sample)
            lines += [f"  ── 서술 프로브 (본 실험과 별개, {record['method']} 디코딩, "
                      f"max {self.probe_max_new_tokens}tok) ──",
                      f"  프로브 Q  : {self.probe_prompt}",
                      f"  전체 출력 : {probe_out['prediction']!r}"]

        # 프레임 + 프로브 저장 (모니터링 전용 파일)
        saved = self._save_artifacts(k, sample, record, probe_out)
        if saved:
            lines.append(f"  저장      : {saved}")
        lines.append("═" * 70)
        logger.info("\n".join(lines))

    def _run_probe(self, method, sample) -> dict:
        probe_sample = dataclasses.replace(sample, question=self.probe_prompt)
        old_max, old_suffix = method.max_new_tokens, method.prompt_suffix
        method.max_new_tokens, method.prompt_suffix = self.probe_max_new_tokens, ""
        try:
            return method.generate(probe_sample)
        finally:
            method.max_new_tokens, method.prompt_suffix = old_max, old_suffix

    def _save_artifacts(self, k: int, sample, record: dict, probe_out) -> str:
        d = self.ckpt_dir / f"ckpt{k}_{record['video_id']}"
        d.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoint": k, "sample_id": record["sample_id"],
                   "question": record["question"], "ground_truth": record["ground_truth"],
                   "prediction": record["prediction"], "internals": record.get("internals", {}),
                   "scene_caption_gt": self._caption(record["video_id"]),
                   "probe_prompt": self.probe_prompt if probe_out else None,
                   "probe_output": probe_out["prediction"] if probe_out else None}
        (d / "inspection.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        n_frames = self._extract_frames(sample.video_path, d)
        return f"{d} (frames {n_frames}장)" if n_frames else str(d)

    def _extract_frames(self, video_path, out_dir: Path) -> int:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not video_path or not os.path.exists(video_path):
            return 0
        ok = 0
        for i in range(self.probe_frames):
            dst = out_dir / f"frame_{i}.jpg"
            r = subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", str(i * 2), "-i",
                                str(video_path), "-frames:v", "1", "-q:v", "3", str(dst)],
                               capture_output=True)
            ok += int(r.returncode == 0 and dst.exists())
        return ok


def out_paths(cfg, benchmark: str, model: str, method: str, dry_run: bool,
              out_tag: str = "") -> tuple[Path, Path]:
    root = Path(cfg.get("paths.results_dir", "results"))
    sub = "dryrun" if dry_run else "runs"
    d = root / sub / benchmark
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{method}" + (f"__{out_tag}" if out_tag else "")
    return d / f"{stem}.jsonl", d / f"{stem}.meta.json"


def load_done_records(jsonl_path: Path) -> list:
    """기존 결과 레코드 로드 (깨진 라인은 무시) — 재개 skip과 모니터 누적치 시드에 사용."""
    records = []
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("깨진 결과 라인 무시: %.80s", line)
    return records


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
    prev_records = load_done_records(jsonl_path)
    if args.retry_errors:
        n_err_drop = sum(1 for r in prev_records if str(r.get("prediction", "")).startswith("ERROR:"))
        if n_err_drop:
            prev_records = [r for r in prev_records
                            if not str(r.get("prediction", "")).startswith("ERROR:")]
            tmp = jsonl_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in prev_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(jsonl_path)
            logger.info("--retry-errors: ERROR 기록 %d건 제거 후 재시도", n_err_drop)
    done = {r["sample_id"] for r in prev_records if "sample_id" in r}
    todo = [s for s in samples if s.sample_id not in done]
    logger.info("[%s×%s×%s] 전체 %d, 완료 %d, 남음 %d → %s",
                args.model, args.method, args.benchmark,
                len(samples), len(samples) - len(todo), len(todo), jsonl_path)
    write_meta(meta_path, args, cfg, n_total=len(samples), n_done=len(samples) - len(todo))

    if not todo:
        logger.info("처리할 샘플 없음 (모두 완료).")
        return 0

    monitor = RunMonitor(cfg, args, n_todo=len(todo), jsonl_path=jsonl_path)
    monitor.seed_from_records(prev_records)   # 재개 시 누적 지표가 전체 기준이 되도록

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
            monitor.observe(record)
            if (i + 1) % args.log_interval == 0:
                rate = (time.time() - t_start) / (i + 1)
                logger.info("  [%d/%d] %.2f s/sample | err %d%s",
                            i + 1, len(todo), rate, n_err, monitor.oneline_suffix())
            monitor.maybe_checkpoint(i + 1, sample, record, method)

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
    p.add_argument("--retry-errors", action="store_true",
                   help="기존 결과의 ERROR 샘플을 제거하고 재시도 (OOM 등 수정 후 재실행용)")
    p.add_argument("--log-interval", type=int, default=50,
                   help="N샘플마다 한 줄 상태(누적acc·예측분포·internals 요약) 출력")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(run(build_parser().parse_args()))
