#!/bin/bash
# fixTable: 34건 오프로드 재시도 2차 — max_memory_gpu 19GiB→13GiB 교정 + 최종 표.
#
# finalS4 [1]단계 실측: 상한 19GiB는 가중치 총량(~16GB)보다 커서 오프로드가 전혀 일어나지
# 않았고(전 레이어 GPU 배치) OOM 재발. 실제 병목은 활성화(~7-8GB)이므로 가중치 상한을
# 13GiB로 내려 ~6개 레이어를 CPU에 배치, GPU에 ~10GB 활성화 여유를 확보한다.
#
# 제출: qsub -N G1C8_hsm_fixTable -W depend=afterany:<finalS4> -v PAYLOAD="scripts/fixTable_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/2] 34건 오프로드 재시도 (GPU 상한 13GiB — 레이어 실제 오프로드)'
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] || continue
  bench=$(basename $(dirname "$f")); model=${stem%%__*}; method=${stem##*__}
  echo "offload13 retry: $model $method $bench ($e errors)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method $method --benchmark $bench --retry-errors \
    --set models.qwen2_5_omni_7b.device_map=auto \
    --set models.qwen2_5_omni_7b.max_memory_gpu=13GiB \
    > logs/fixT_${model}_${method}_${bench}.log 2>&1 || true
done

echo '[2/2] 최종 채점·집계 — 목표(1) 최종 표 (에러 0 판)'
conda run -n videollama2 python -m src.score --jsonl 'results/runs/avhbench/*.jsonl' 'results/runs/cmm/*.jsonl' > logs/final_score.log 2>&1 || true
conda run -n videollama2 python -m src.aggregate --results results/runs > logs/final_table.log 2>&1 || true
echo '  잔여 에러 수 (0이어야 함):'
REMAIN=0
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] && { echo "  $f: $e"; REMAIN=$((REMAIN+e)); }
done
[ "$REMAIN" -eq 0 ] && echo '  전부 0 — 에러 0 달성'
echo "FIXTABLE-DONE (잔여에러=$REMAIN)"
