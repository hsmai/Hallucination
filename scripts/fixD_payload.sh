#!/bin/bash
# fixD: 재시도 2패스로도 안 풀린 24GB 초과 클립(Qwen 34개) — CPU 오프로드로 에러 0화.
#
# 배경: retryA에서 Qwen AVCD(AVH 17/CMM 13), MAD CMM 2, VCD CMM 2가 프레시 로드에도
# 1.5~1.6GiB 할당 실패 → 클립 자체가 24GB 한계 초과. 같은 방식 재시도는 무의미하므로
# device_map=auto + max_memory(GPU 19GiB 상한)로 초과 레이어를 CPU에 배치해 재실행한다.
# (src/adapters/qwen_omni.py의 --set models.qwen2_5_omni_7b.device_map=auto 경로.
#  기본 로드 경로는 무변경 — 이 override 없이는 기존과 동일하게 device_map="cuda".)
#
# 제출: qsub -N G1C8_hsm_fixD -W depend=afterany:<steerC_jobid> \
#         -v PAYLOAD="scripts/fixD_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }
echo '[D0] 정성 샘플 패키징 (regenV2 산출물 병합 — CPU만, 오프로드 재시도보다 먼저)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/avhbench__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/AVH_Bench \
    --benchmark avhbench --top 15 \
    --model ${model} --regen-dir results/runs/avhbench \
    --out results/runs/qualitative/avhbench__${model} \
    > logs/package_${model}.log 2>&1 || true
done
echo '[D0] 패키징 완료 — results/runs/qualitative/'
echo '[D] 오프로드 재시도 (24GB 초과 클립)'
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] || continue
  bench=$(basename $(dirname "$f")); model=${stem%%__*}; method=${stem##*__}
  echo "fixD: $model $method $bench ($e errors)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method $method --benchmark $bench --retry-errors \
    --set models.qwen2_5_omni_7b.device_map=auto \
    > logs/fixD_${model}_${method}_${bench}.log 2>&1 || true
done
echo '[D] 최종 채점·집계·마이닝 재실행'
conda run -n videollama2 python -m src.score --jsonl 'results/runs/avhbench/*.jsonl' 'results/runs/cmm/*.jsonl' > logs/final_score.log 2>&1 || true
conda run -n videollama2 python -m src.aggregate --results results/runs > logs/final_table.log 2>&1 || true
for model in videollama2_av qwen2_5_omni_7b; do for bench in avhbench cmm; do
  conda run -n videollama2 python -m src.mining --results results/runs --benchmark $bench --model $model >> logs/final_mining.log 2>&1 || true
done; done
echo '[D] 잔여 에러 수 (0이어야 함):'
REMAIN=0
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] && { echo "  $f: $e"; REMAIN=$((REMAIN+e)); }
done
[ "$REMAIN" -eq 0 ] && echo '  전부 0 — 에러 0 달성'
echo D-DONE
