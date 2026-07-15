#!/usr/bin/env bash
# 16개 조합(2모델 × 4방법 × 2벤치마크) 전부 MockModel dry-run → 채점 → 집계.
# 로컬 전용 배관 검증. 재실행 안전 (처리분 skip).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LIMIT="${1:-0}"   # 인자로 샘플 수 제한 가능 (기본 0 = 전체)

for model in videollama2_av qwen2_5_omni_7b; do
  for method in base vcd_ext mad avcd; do
    for bench in avhbench cmm; do
      echo "=== dry-run: $model × $method × $bench ==="
      python3 -m src.runner --model "$model" --method "$method" --benchmark "$bench" \
        --dry-run ${LIMIT:+--limit $LIMIT} --log-interval 2000
    done
  done
done

echo "=== 채점 ==="
python3 -m src.score --jsonl "results/dryrun/avhbench/*.jsonl" "results/dryrun/cmm/*.jsonl"

echo "=== 집계 (dry-run이므로 게이트 대조 생략) ==="
python3 -m src.aggregate --results results/dryrun --no-gate

echo "=== D3 마이닝 (dry-run 데이터로 배관 확인) ==="
python3 -m src.mining --results results/dryrun --benchmark avhbench --model videollama2_av

echo "OK: 전 구간 통과"
