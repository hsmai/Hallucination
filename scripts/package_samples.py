#!/usr/bin/env python3
"""D3 패키징 — 마이닝 후보를 샘플당 폴더 하나로: 대표 프레임 + 질문/방법별 답변.

사용 (서버, S4):
  python scripts/package_samples.py \
      --mining results/runs/mining/avhbench__videollama2_av.csv \
      --media-dir /path/to/AVHBench \
      --out results/runs/qualitative/avhbench__videollama2_av \
      [--top 30] [--frames 6]

출력 구조:
  {out}/{video_id}__{n}/
    frame_0.jpg ... frame_5.jpg   (ffmpeg 균등 샘플링)
    sample.json                   (질문/GT/방법별 답변/internals)
    answers.txt                   (사람이 바로 읽는 요약)

서술형 답변 재생성(선택): 후보에 대해 러너를 다시 돌려 긴 답변 확보 후 재패키징
  python -m src.runner --model M --method base --benchmark B \
      --ids-file results/runs/mining/{stem}.ids.txt --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("에러: ffmpeg가 PATH에 없습니다. 서버에서 `module load ffmpeg` 또는 conda로 설치 후 재실행하세요.")
    return exe


def video_duration(path: Path) -> float:
    probe = shutil.which("ffprobe")
    if not probe:
        return 0.0
    try:
        out = subprocess.run(
            [probe, "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, check=True).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def extract_frames(ffmpeg: str, video: Path, out_dir: Path, n_frames: int) -> int:
    dur = video_duration(video)
    ok = 0
    for i in range(n_frames):
        # 균등 샘플링 (duration을 모르면 i초 지점 시도)
        ts = (dur * (i + 0.5) / n_frames) if dur > 0 else float(i)
        dst = out_dir / f"frame_{i}.jpg"
        r = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-ss", f"{ts:.2f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(dst)],
            capture_output=True, text=True)
        if r.returncode == 0 and dst.exists():
            ok += 1
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="D3 정성 샘플 패키징")
    ap.add_argument("--mining", required=True, help="src.mining이 만든 csv")
    ap.add_argument("--media-dir", required=True, help="비디오 루트 (AVHBench면 videos/ 상위)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30, help="상위 N개 후보만 (csv는 이미 강한 순 정렬)")
    ap.add_argument("--frames", type=int, default=6, help="샘플당 프레임 수 (4~8)")
    ap.add_argument("--benchmark", default="avhbench", choices=("avhbench", "cmm"))
    ap.add_argument("--qa-json", default="data/qa/avhbench_qa.json",
                    help="AV Captioning GT 조회용 (그림의 Sound/Video 설명 주석 재료, avhbench만)")
    ap.add_argument("--model", default=None,
                    help="서술형(free-form) jsonl 병합용 모델 키 (--regen-dir와 함께)")
    ap.add_argument("--regen-dir", default=None,
                    help="{model}__{method}__free*.jsonl 디렉터리 (예: results/runs/avhbench)")
    args = ap.parse_args()

    # regenV2 서술형 출력 병합: method -> {sample_id: 긴 raw text}
    free_text: dict = {}
    if args.model and args.regen_dir:
        for m in ("base", "vcd_ext", "mad", "avcd"):
            d = free_text.setdefault(m, {})
            for p in sorted(Path(args.regen_dir).glob(f"{args.model}__{m}__free*.jsonl")):
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    d[rec["sample_id"]] = rec.get("prediction") or ""
        n_free = sum(len(v) for v in free_text.values())
        if n_free == 0:
            print("경고: free-form jsonl을 찾지 못함 — 단답만 패키징됩니다", file=sys.stderr)

    # 장면+소리 설명 GT (MAD 그림의 'Sound:/Video:' 주석 역할, 커버리지 ~79%)
    captions = {}
    if args.benchmark == "avhbench" and Path(args.qa_json).exists():
        qa = json.loads(Path(args.qa_json).read_text())
        captions = {x["video_id"]: x["label"] for x in qa if x.get("task") == "AV Captioning"}

    ffmpeg = check_ffmpeg()
    media = Path(args.media_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.mining, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.top]
    if not rows:
        sys.exit(f"에러: 마이닝 csv가 비어 있습니다: {args.mining}")

    n_ok = 0
    for i, r in enumerate(rows):
        vid = r["video_id"]
        if args.benchmark == "avhbench":
            video = media / "videos" / f"{vid}.mp4"
        else:
            video = None  # CMM은 csv에 없는 원경로 필요 — sample.json의 비디오는 스킵 가능
            for cand in media.rglob(f"{vid}.mp4"):
                video = cand
                break
        folder = out_root / f"{vid}__{i:03d}"
        folder.mkdir(parents=True, exist_ok=True)

        got = 0
        if video and video.exists():
            got = extract_frames(ffmpeg, video, folder, args.frames)
        else:
            print(f"경고: 비디오 없음 — {vid} (프레임 생략)", file=sys.stderr)

        r["scene_description_gt"] = captions.get(vid, "")
        sid = r["sample_id"]
        for m, d in free_text.items():
            if sid in d:
                r[f"{m}_free_text"] = d[sid]
        (folder / "sample.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        free_block = ""
        if free_text:
            names = {"base": "Base", "vcd_ext": "VCD-ext", "mad": "MAD", "avcd": "AVCD"}
            lines = [f"{names[m]:7s}: {free_text[m][sid]}"
                     for m in ("base", "vcd_ext", "mad", "avcd")
                     if sid in free_text.get(m, {})]
            if lines:
                free_block = ("\n— 서술형 raw text (suffix 없는 자유 응답, MAD Fig.9-10 방식) —\n"
                              + "\n".join(lines) + "\n")
        (folder / "answers.txt").write_text(
            (f"장면 설명(GT 캡션): {r['scene_description_gt']}\n\n" if r["scene_description_gt"] else "")
            + f"Q: {r['question']}\nGT: {r['ground_truth']}\n\n"
            f"Base   : {r.get('base_pred','')}  (correct={r.get('base_correct','')})\n"
            f"VCD-ext: {r.get('vcd_ext_pred','')}  (correct={r.get('vcd_ext_correct','')})\n"
            f"MAD    : {r.get('mad_pred','')}  (오답)\n"
            f"AVCD   : {r.get('avcd_pred','')}  (오답)\n"
            f"OURS   : (선배 결과 join 후 기입)\n"
            + free_block)
        n_ok += 1
        print(f"[{i+1}/{len(rows)}] {vid}: frames={got} → {folder}")

    print(f"\n완료: {n_ok}개 샘플 패키징 → {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
