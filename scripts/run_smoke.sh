#!/usr/bin/env bash
# S1 스모크: 16조합 × 5샘플 — 실모델 배관 첫 검증 (서버 전용).
# 사용: bash scripts/run_smoke.sh [샘플수=5]
# 환경: conda env 2종을 모델별로 자동 선택 (아래 변수로 조정).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N="${1:-5}"
QWEN_ENV="${QWEN_ENV:-qwen-omni}"
VL_ENV="${VL_ENV:-videollama2}"
RESULTS="${RESULTS:-results/smoke}"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"

env_for() { [ "$1" = "qwen2_5_omni_7b" ] && echo "$QWEN_ENV" || echo "$VL_ENV"; }

FAIL=0
for model in videollama2_av qwen2_5_omni_7b; do
  for bench in avhbench cmm; do
    for method in base vcd_ext mad avcd; do
      tag="${model}×${method}×${bench}"
      log="$LOG_DIR/smoke_${model}_${method}_${bench}.log"
      echo "=== smoke: $tag (env: $(env_for $model)) ==="
      if conda run -n "$(env_for $model)" --no-capture-output \
          python -m src.runner --model "$model" --method "$method" --benchmark "$bench" \
          --limit "$N" --set paths.results_dir="$RESULTS" >"$log" 2>&1; then
        echo "  OK"
      else
        echo "  ✗ FAIL — $log 확인"; FAIL=$((FAIL+1))
      fi
      # VRAM 스냅샷 (모델 로드 흔적)
      nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader >> "$log" 2>/dev/null || true
    done
  done
done

echo; echo "=== 스모크 채점 (형식 확인용) ==="
conda run -n "$VL_ENV" --no-capture-output \
  python -m src.score --jsonl "$RESULTS/runs/avhbench/*.jsonl" "$RESULTS/runs/cmm/*.jsonl" || true

if [ "$FAIL" -gt 0 ]; then
  echo "✗ 실패 $FAIL건 — logs/ 에서 원인 확인 후 runbook '트러블슈팅' 참조"; exit 1
fi
echo "OK: 16조합 스모크 전부 통과. 각 로그 말미의 VRAM·속도로 소요시간표를 갱신하세요."
