"""집계기 — 채점된 JSONL들 → D1 정량 테이블 (md/csv) + 카테고리 세부표 + 게이트 대조.

메인 테이블 형식 = MAD 논문 Table 1 (Experiments.pdf 상단 표):
  행: model × method, 열: CMM(Visual/Audio/Language Dom, Overall) + AVHBench(VdAH, AdVH, Overall)

CLI: python -m src.aggregate --results results/dryrun [--out results/dryrun/tables]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from .data import AVH_TASK_ADVH, AVH_TASK_VDAH

# CMM sub_category → 논문 표 열 이름
CMM_COLS = {
    "overrely_visual_ignore_audio": "Visual Dom.",
    "overrely_audio_ignore_visual": "Audio Dom.",
    "overrely_language_ignore_visual": "Language Dom.",
}
AVH_COLS = {
    AVH_TASK_VDAH: "Video-Driven Audio Hall.",
    AVH_TASK_ADVH: "Audio-Driven Video Hall.",
}
METHOD_ORDER = ["base", "vcd_ext", "avcd", "mad"]
MODEL_ORDER = ["videollama2_av", "qwen2_5_omni_7b"]
COLUMNS = (["Visual Dom.", "Audio Dom.", "Language Dom.", "CMM Overall"]
           + ["Video-Driven Audio Hall.", "Audio-Driven Video Hall.", "AVH Overall"])


def read_scored(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    unscored = sum(1 for r in out
                   if r["correct"] is None and not r["prediction"].startswith("ERROR:"))
    if unscored:
        raise ValueError(f"{path}: 채점 안 된 레코드 {unscored}건 — 먼저 python -m src.score 실행")
    return out


def acc(records: list[dict], overall_tasks: set | None = None) -> dict:
    """카테고리별 + overall 정확도. overall_tasks 지정 시 그 카테고리만 overall에 포함."""
    by_cat = defaultdict(lambda: [0, 0])
    for r in records:
        if r["correct"] is None:
            continue
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += int(r["correct"])
    cats = {k: c / n * 100 for k, (c, n) in by_cat.items() if n}
    pool = [(c, n) for k, (c, n) in by_cat.items()
            if overall_tasks is None or k in overall_tasks]
    tc, tn = sum(c for c, _ in pool), sum(n for _, n in pool)
    return {"cats": cats, "overall": tc / tn * 100 if tn else None, "n": tn}


def collect(results_dir: Path, cfg) -> dict:
    """(model, method) → {열이름: acc}. results/{runs|dryrun}/{benchmark}/{model}__{method}.jsonl"""
    overall_tasks = set(cfg.get("benchmarks.avhbench.tasks_for_overall"))
    table = defaultdict(dict)
    counts = defaultdict(dict)
    for bench, colmap in (("cmm", CMM_COLS), ("avhbench", AVH_COLS)):
        d = results_dir / bench
        if not d.exists():
            continue
        for p in sorted(d.glob("*__*.jsonl")):
            model, method = p.stem.split("__", 1)
            if "__" in method:
                continue  # 재생성 등 태그 붙은 파일(model__method__tag)은 본 테이블에서 제외
            recs = read_scored(p)
            a = acc(recs, overall_tasks if bench == "avhbench" else None)
            for cat, col in colmap.items():
                if cat in a["cats"]:
                    table[(model, method)][col] = a["cats"][cat]
            key = "CMM Overall" if bench == "cmm" else "AVH Overall"
            table[(model, method)][key] = a["overall"]
            counts[(model, method)][key] = a["n"]
    return {"table": dict(table), "counts": dict(counts)}


def fmt(v) -> str:
    return f"{v:.1f}" if isinstance(v, float) else "—"


def render_markdown(table: dict, counts: dict, gate: dict | None, title: str) -> str:
    lines = [f"# {title}", "", "| Model | Method | " + " | ".join(COLUMNS) + " |",
             "|---|---|" + "---|" * len(COLUMNS)]
    keys = sorted(table.keys(), key=lambda k: (
        MODEL_ORDER.index(k[0]) if k[0] in MODEL_ORDER else 99,
        METHOD_ORDER.index(k[1]) if k[1] in METHOD_ORDER else 99))
    for model, method in keys:
        row = table[(model, method)]
        lines.append(f"| {model} | {method} | "
                     + " | ".join(fmt(row.get(c)) for c in COLUMNS) + " |")
    lines.append("")
    n_note = {f"{m}/{me}": c for (m, me), c in counts.items()}
    lines.append(f"_샘플 수(overall 기준): {json.dumps(n_note, ensure_ascii=False)}_")

    if gate:
        lines += ["", "## 게이트 대조 (gate_targets, 허용 ±{:.1f}%p)".format(gate["tol"]), "",
                  "| Model | Method | 항목 | 우리 | 목표 | Δ | 판정 |", "|---|---|---|---|---|---|---|"]
        for row in gate["rows"]:
            lines.append("| {model} | {method} | {item} | {ours} | {target} | {delta} | {verdict} |"
                         .format(**row))
    return "\n".join(lines) + "\n"


def gate_compare(table: dict, cfg) -> dict:
    """MAD 논문 Table 1 + Ours(Base)와의 자동 대조."""
    tol = float(cfg.get("gate_targets.tolerance_pp"))
    paper = cfg.get("gate_targets.mad_paper_table1")
    col_order_cmm = ["Visual Dom.", "Audio Dom.", "Language Dom.", "CMM Overall"]
    col_order_avh = ["Video-Driven Audio Hall.", "Audio-Driven Video Hall.", "AVH Overall"]
    rows = []
    for model, methods in paper.items():
        for method, targets in methods.items():
            ours_row = table.get((model, method))
            if not ours_row:
                continue
            for cols, vals in ((col_order_cmm, targets.get("cmm")),
                               (col_order_avh, targets.get("avh"))):
                if not vals:
                    continue
                for col, target in zip(cols, vals):
                    ours = ours_row.get(col)
                    if ours is None:
                        continue
                    delta = ours - target
                    ok = abs(delta) <= tol
                    note = " (참고)" if method == "avcd" else ""  # 저자 재구현 수치 — 참고 기준
                    rows.append({"model": model, "method": method, "item": col,
                                 "ours": f"{ours:.1f}", "target": f"{target:.1f}{note}",
                                 "delta": f"{delta:+.1f}",
                                 "verdict": ("PASS" if ok else "FAIL") + note})
    return {"tol": tol, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="JSONL → D1 정량 테이블")
    ap.add_argument("--results", default="results/runs", help="results/runs 또는 results/dryrun")
    ap.add_argument("--out", default=None, help="기본: {results}/tables")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-gate", action="store_true", help="게이트 대조 생략 (dry-run용)")
    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config(args.config)

    results_dir = Path(args.results)
    got = collect(results_dir, cfg)
    if not got["table"]:
        print(f"결과 JSONL이 없습니다: {results_dir}/{{cmm,avhbench}}/*.jsonl", file=sys.stderr)
        return 1

    gate = None if args.no_gate else gate_compare(got["table"], cfg)
    title = f"D1 정량 테이블 — {results_dir}"
    md = render_markdown(got["table"], got["counts"], gate, title)

    out_dir = Path(args.out) if args.out else results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_table.md").write_text(md)

    with open(out_dir / "main_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method"] + COLUMNS)
        for (model, method), row in sorted(got["table"].items()):
            w.writerow([model, method] + [row.get(c, "") for c in COLUMNS])

    print(md)
    print(f"저장: {out_dir}/main_table.md, main_table.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
