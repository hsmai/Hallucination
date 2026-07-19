#!/bin/bash
# fixCPU: AVCD 30건(24GB 초과 클립) 완전 CPU 재실행 → 에러 0 (동일 조건 유지).
#
# 배경: 이 30건은 base/MAD에선 성공하나 AVCD만 실패 — AVCD는 스텝마다 원본+마스킹 3회
# re-forward + eager attention [H,N,N]이라 긴 클립 활성화가 24GB 초과. 오프로드(가중치만
# 절약)로는 불가. device_map=cpu로 활성화까지 CPU RAM(fp32)에서 계산 → 프레임/수식 무변경.
# GPU 노드에서 돌리되(models.py의 CUDA 가용성 체크 통과용) 실연산은 CPU.
#
# 제출: qsub -N G1C8_hsm_fixCPU -v PAYLOAD="scripts/fixCPU_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
export OMP_NUM_THREADS=8                 # ncpus=8 활용 (CPU forward 병렬)
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/2] 잔여 에러 CPU 재실행 (device_map=cpu, fp32)'
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] || continue
  bench=$(basename $(dirname "$f")); model=${stem%%__*}; method=${stem##*__}
  echo "cpu retry: $model $method $bench ($e errors)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method $method --benchmark $bench --retry-errors \
    --set models.qwen2_5_omni_7b.device_map=cpu \
    > logs/fixCPU_${model}_${method}_${bench}.log 2>&1 || true
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
[ "$REMAIN" -eq 0 ] && echo '  ✅ 전부 0 — 에러 0 달성'
echo "FIXCPU-DONE (잔여에러=$REMAIN)"
