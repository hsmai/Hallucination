#!/bin/bash
# finalS4: S4 마무리 일괄 — (1) 34건 오프로드 재시도→에러0 (2) 최종 표 (3) regenV3 probe 서술형 (4) 패키징
#
# 선행 실패 교정 2건:
#  - fixD: --set models.qwen2_5_omni_7b.device_map가 yaml에 키가 없어 KeyError → 키 추가 후 재시도
#  - regenV2: suffix만 제거하면 VL2가 여전히 단답("No.") → 질문 자체를 monitoring.probe_prompt로
#    치환하는 --use-probe-question 사용 (S3 체크포인트 probe에서 VL2 전 방법 긴 서술 실증됨)
#
# 제출: qsub -N G1C8_hsm_finalS4 -v PAYLOAD="scripts/finalS4_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/4] 34건 오프로드 재시도 (device_map=auto, GPU 19GiB 상한)'
for f in results/runs/*/*.jsonl; do
  stem=$(basename "$f" .jsonl)
  case "$stem" in *__*__*) continue;; esac
  e=$(grep -c '"ERROR' "$f" 2>/dev/null || true)
  [ "$e" -gt 0 ] || continue
  bench=$(basename $(dirname "$f")); model=${stem%%__*}; method=${stem##*__}
  echo "offload retry: $model $method $bench ($e errors)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method $method --benchmark $bench --retry-errors \
    --set models.qwen2_5_omni_7b.device_map=auto \
    > logs/fixD2_${model}_${method}_${bench}.log 2>&1 || true
done

echo '[2/4] 최종 채점·집계 — 목표(1) 최종 표'
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
echo "TABLE-READY (잔여에러=$REMAIN)"

echo '[3/4] regenV3 — probe 프롬프트 서술형 재생성 (MAD Fig.9-10 방식)'
for model in videollama2_av qwen2_5_omni_7b; do
  ids=logs/regen_ids_${model}.txt
  ids_avcd=logs/regen_ids_avcd_${model}.txt
  [ -f "$ids" ] || { echo "skip: $ids 없음"; continue; }
  for m in base vcd_ext mad; do
    echo "regenV3: $model $m (probe 192tok)"
    conda run -n $(env_for $model) --no-capture-output python -m src.runner \
      --model $model --method $m --benchmark avhbench \
      --ids-file $ids --max-new-tokens 192 --out-tag free192v2 \
      --use-probe-question --set monitoring.enabled=false \
      > logs/regen3_${model}_${m}.log 2>&1 || true
  done
  echo "regenV3: $model avcd (probe 96tok)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark avhbench \
    --ids-file $ids_avcd --max-new-tokens 96 --out-tag free96v2 \
    --use-probe-question --set monitoring.enabled=false \
    > logs/regen3_${model}_avcd.log 2>&1 || true
done
echo '[3/4] 출력 길이 확인 (med가 한 자릿수면 실패):'
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob('results/runs/avhbench/*__free*v2.jsonl')):
    lens = [len((json.loads(l).get('prediction') or '')) for l in open(f)]
    if lens:
        print(f, 'n=', len(lens), 'len min/med/max=',
              min(lens), sorted(lens)[len(lens)//2], max(lens))
EOF

echo '[4/4] 정성 샘플 패키징 (v2 서술형 병합 — glob 정렬상 v2가 나중이라 v2 우선)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/avhbench__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/AVH_Bench \
    --benchmark avhbench --top 15 \
    --model ${model} --regen-dir results/runs/avhbench \
    --out results/runs/qualitative/avhbench__${model} \
    > logs/package2_${model}.log 2>&1 || true
done
echo FINAL-S4-DONE
