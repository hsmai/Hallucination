"""D3 사전 마이닝 — "MAD와 AVCD가 동시에 오답인 샘플" 목록 (+Base 오답 태그).

선배는 이 목록에서 OURS 정답 여부만 확인하면 논문 Figure 후보가 완성된다 (blueprint D3).

CLI: python -m src.mining --results results/runs --benchmark avhbench --model videollama2_av
     → results/{...}/mining/{benchmark}__{model}.md + .csv + ids.txt
       (ids.txt는 러너 --ids-file --max-new-tokens 재생성 입력으로 사용)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_by_id(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def mine(results_dir: Path, benchmark: str, model: str) -> list[dict]:
    d = results_dir / benchmark
    need = {}
    for method in ("base", "mad", "avcd", "vcd_ext"):
        p = d / f"{model}__{method}.jsonl"
        if method in ("mad", "avcd") and not p.exists():
            raise FileNotFoundError(f"마이닝에 필수인 결과가 없습니다: {p}")
        need[method] = load_by_id(p) if p.exists() else {}

    candidates = []
    for sid, mad_r in need["mad"].items():
        avcd_r = need["avcd"].get(sid)
        if avcd_r is None:
            continue
        if mad_r["correct"] is None or avcd_r["correct"] is None:
            continue  # ERROR 샘플 제외
        if mad_r["correct"] or avcd_r["correct"]:
            continue  # 둘 다 오답만
        base_r = need["base"].get(sid)
        vcd_r = need["vcd_ext"].get(sid)
        candidates.append({
            "sample_id": sid,
            "video_id": mad_r["video_id"],
            "category": mad_r["category"],
            "question": mad_r["question"],
            "ground_truth": mad_r["ground_truth"],
            "base_pred": base_r["prediction"] if base_r else "",
            "base_correct": base_r["correct"] if base_r else None,
            "vcd_ext_pred": vcd_r["prediction"] if vcd_r else "",
            "vcd_ext_correct": vcd_r["correct"] if vcd_r else None,
            "mad_pred": mad_r["prediction"],
            "avcd_pred": avcd_r["prediction"],
            "mad_internals": mad_r.get("internals", {}),
            "avcd_internals": avcd_r.get("internals", {}),
        })
    # base까지 오답인 샘플을 앞으로 (전 방법 오답 = OURS만 맞으면 가장 강한 Figure)
    candidates.sort(key=lambda c: (c["base_correct"] is not False, c["sample_id"]))
    return candidates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D3 마이닝: MAD·AVCD 동시 오답 샘플")
    ap.add_argument("--results", default="results/runs")
    ap.add_argument("--benchmark", required=True, choices=("avhbench", "cmm"))
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    cands = mine(results_dir, args.benchmark, args.model)

    out_dir = results_dir / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.benchmark}__{args.model}"

    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["sample_id", "video_id", "category", "question", "ground_truth",
                "base_correct", "base_pred", "vcd_ext_correct", "vcd_ext_pred",
                "mad_pred", "avcd_pred"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)

    (out_dir / f"{stem}.ids.txt").write_text(
        "\n".join(c["sample_id"] for c in cands) + ("\n" if cands else ""))

    lines = [f"# D3 마이닝 — {args.benchmark} × {args.model}",
             "", f"MAD·AVCD 동시 오답: **{len(cands)}건** "
             f"(그중 Base도 오답: {sum(1 for c in cands if c['base_correct'] is False)}건 — 상단 정렬)",
             "", "| sample_id | category | GT | Base | VCD-ext | MAD | AVCD |", "|---|---|---|---|---|---|---|"]
    for c in cands[:200]:
        mark = lambda ok, pred: f"{'✗' if ok is False else '✓' if ok else '?'} {pred[:20]}"
        lines.append(f"| {c['sample_id'][:60]} | {c['category'][:20]} | {c['ground_truth']} "
                     f"| {mark(c['base_correct'], c['base_pred'])} "
                     f"| {mark(c['vcd_ext_correct'], c['vcd_ext_pred'])} "
                     f"| ✗ {c['mad_pred'][:20]} | ✗ {c['avcd_pred'][:20]} |")
    if len(cands) > 200:
        lines.append(f"\n_(상위 200건만 표시 — 전체는 csv 참조)_")
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")

    print(f"후보 {len(cands)}건 → {out_dir}/{stem}.{{md,csv,ids.txt}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
