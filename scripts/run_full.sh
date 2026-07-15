#!/usr/bin/env bash
# S3 본 실행 (서버 전용): 16 run 전체 — 우선순위 순, 완료분부터 즉시 채점.
# 사용(야간 무인): mkdir -p logs && nohup bash scripts/run_full.sh > logs/full_$(date +%m%d_%H%M).log 2>&1 &
# 중단-재개 안전: 재실행하면 처리된 sample_id는 자동 skip.
#
# 우선순위 (blueprint S3 — GPU 시간 부족 시 아래쪽부터 포기):
#   P1. AVHBench 전 방법 (정성 샘플의 주 무대) — 모델별 묶음
#   P2. CMM base/MAD/AVCD
#   P3. CMM VCD-ext
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

QWEN_ENV="${QWEN_ENV:-qwen-omni}"
VL_ENV="${VL_ENV:-videollama2}"
RESULTS="${RESULTS:-results}"           # 본 실행은 기본 results/runs
LOG_DIR="logs"; mkdir -p "$LOG_DIR"

env_for() { [ "$1" = "qwen2_5_omni_7b" ] && echo "$QWEN_ENV" || echo "$VL_ENV"; }

one_run() { # model method bench
  local model="$1" method="$2" bench="$3"
  local log="$LOG_DIR/full_${model}_${method}_${bench}.log"
  echo "[$(date +%H:%M:%S)] RUN $model × $method × $bench"
  if conda run -n "$(env_for $model)" --no-capture-output \
      python -m src.runner --model "$model" --method "$method" --benchmark "$bench" \
      --set paths.results_dir="$RESULTS" >>"$log" 2>&1; then
    # 완료 즉시 채점 (본 실행과 병행)
    conda run -n "$(env_for $model)" --no-capture-output \
      python -m src.score --jsonl "$RESULTS/runs/$bench/${model}__${method}.jsonl" \
      >>"$log" 2>&1 || true
    echo "[$(date +%H:%M:%S)]   done."
  else
    echo "[$(date +%H:%M:%S)]   ✗ FAIL — $log (재실행 시 이어서 처리됨)"
  fi
}

# ---- P1: AVHBench 전 방법 (모델 단위 묶음 — 같은 모델 4run 연속으로 디스크 캐시 활용)
for model in videollama2_av qwen2_5_omni_7b; do
  for method in base mad avcd vcd_ext; do
    one_run "$model" "$method" avhbench
  done
done

# ---- P2: CMM base/MAD/AVCD
for model in videollama2_av qwen2_5_omni_7b; do
  for method in base mad avcd; do
    one_run "$model" "$method" cmm
  done
done

# ---- P3: CMM VCD-ext
for model in videollama2_av qwen2_5_omni_7b; do
  one_run "$model" vcd_ext cmm
done

echo "########## 최종 집계 ##########"
conda run -n "$VL_ENV" --no-capture-output python -m src.aggregate --results "$RESULTS/runs" \
  | tee "$LOG_DIR/full_final_table.log"

echo "########## D3 마이닝 ##########"
for model in videollama2_av qwen2_5_omni_7b; do
  for bench in avhbench cmm; do
    conda run -n "$VL_ENV" --no-capture-output \
      python -m src.mining --results "$RESULTS/runs" --benchmark "$bench" --model "$model" || true
  done
done

echo "완료. 다음: docs/server_runbook.md §S4 (패키징·OURS join·커밋)"
