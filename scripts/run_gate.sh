#!/usr/bin/env bash
# S2 정합성 게이트 (서버 전용): 방법별 부분 샘플 실행 → 목표치 자동 대조.
#   1) Base ≈ Ours(Base) — 세팅 통일 핵심 검증
#   2) MAD 논문 Table 1 ±2%p 대조 (AVCD 행은 참고 기준)
#   3) AVCD CMM용 α 그리드 (0.5~3.0 step 0.5, 100샘플)
#   4) AVCD β 판정: AVHBench val(205)에서 faithful(β=0.2) vs paper(β=0.1) → 81.95 재현 확인
# 사용: bash scripts/run_gate.sh [샘플수=200]
# ⚠ 부분 샘플이라 ±수 %p 표본 노이즈 존재 — 게이트는 방향성 확인용, 최종 대조는 본 실행.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N="${1:-200}"
GRID_N="${GRID_N:-100}"
QWEN_ENV="${QWEN_ENV:-qwen-omni}"
VL_ENV="${VL_ENV:-videollama2}"
RESULTS="results/gate"
LOG_DIR="logs"; mkdir -p "$LOG_DIR"

env_for() { [ "$1" = "qwen2_5_omni_7b" ] && echo "$QWEN_ENV" || echo "$VL_ENV"; }
runner() { # model method bench extra...
  local model="$1" method="$2" bench="$3"; shift 3
  conda run -n "$(env_for $model)" --no-capture-output \
    python -m src.runner --model "$model" --method "$method" --benchmark "$bench" \
    --set paths.results_dir="$RESULTS" "$@"
}

echo "########## [1/4] 16조합 × ${N}샘플 ##########"
for model in videollama2_av qwen2_5_omni_7b; do
  for bench in avhbench cmm; do
    for method in base vcd_ext mad avcd; do
      echo "--- gate: $model × $method × $bench"
      if [ "$bench" = "cmm" ]; then
        runner "$model" "$method" cmm --ids-file data/qa/gate_cmm_ids.txt --retry-errors \
          2>&1 | tee "$LOG_DIR/gate_${model}_${method}_${bench}.log" | tail -2
      else
        runner "$model" "$method" avhbench --limit "$N" --retry-errors \
          2>&1 | tee "$LOG_DIR/gate_${model}_${method}_${bench}.log" | tail -2
      fi
    done
  done
done

echo "########## 채점 + 게이트 대조표 ##########"
conda run -n "$VL_ENV" --no-capture-output python -m src.score \
  --jsonl "$RESULTS/runs/avhbench/*.jsonl" "$RESULTS/runs/cmm/*.jsonl"
conda run -n "$VL_ENV" --no-capture-output python -m src.aggregate \
  --results "$RESULTS/runs" | tee "$LOG_DIR/gate_table.log"

echo "########## [3/4] AVCD CMM α 그리드 (${GRID_N}샘플) ##########"
for alpha in 0.5 1.0 1.5 2.0 2.5 3.0; do
  for model in videollama2_av qwen2_5_omni_7b; do
    echo "--- α=$alpha $model"
    runner "$model" avcd cmm --ids-file data/qa/gate_cmm_ids.txt \
      --set methods.avcd.alpha.cmm="$alpha" --out-tag "alpha${alpha}" \
      --set monitoring.enabled=false \
      2>&1 | tail -1
  done
done
conda run -n "$VL_ENV" --no-capture-output python -m src.score \
  --jsonl "$RESULTS/runs/cmm/*__alpha*.jsonl" | tee "$LOG_DIR/gate_alpha_grid.log"
echo ">>> α 그리드 결과에서 최고 acc의 α를 configs/unified_settings.yaml methods.avcd.alpha.cmm에 확정 기록"

echo "########## [4/4] AVCD β 판정 (AVHBench val 205, 목표 81.95) ##########"
for mode in true false; do
  echo "--- faithful_mode=$mode (true=β0.2·-1e-4 / false=β0.1·-inf)"
  runner videollama2_av avcd avhbench \
    --set benchmarks.avhbench.split=avcd_val \
    --set methods.avcd.faithful_mode="$mode" --out-tag "beta_faithful_${mode}" \
    --set monitoring.enabled=false \
    2>&1 | tail -1
done
conda run -n "$VL_ENV" --no-capture-output python -m src.score \
  --jsonl "$RESULTS/runs/avhbench/*__beta_*.jsonl" | tee "$LOG_DIR/gate_beta.log"
echo ">>> 81.95에 가까운 모드를 yaml methods.avcd.faithful_mode에 확정 기록"
echo ">>> VCD-ext α 판정: gate_table.log의 vcd_ext 행이 Table 1과 크게 어긋나면"
echo "    --set methods.vcd_ext.alpha=2.5 로 [1/4]의 vcd_ext만 재실행하여 비교 (runbook §S2 참조)"

echo "게이트 완료. logs/gate_table.log의 PASS/FAIL과 α/β 판정을 검토 후 yaml을 갱신·커밋하세요."
