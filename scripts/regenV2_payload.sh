#!/bin/bash
# regenV2: 정성 샘플 서술형 재생성 (B 재실행 — 프롬프트 교정판).
#
# 배경: chain_payload(B)는 max_new_tokens만 늘리고 yes/no 강제 suffix를 그대로 둬서
# 출력이 "Yes"/"No" 단답으로 끝났다 (EOS 즉시 종료 — regen192 jsonl len 2~3 확인).
# MAD Fig.9-10 스타일 raw text를 위해 suffix를 비우고(--set prompts.avhbench_suffix.*="")
# 같은 마이닝 ids(top-15 / AVCD top-6)로 자유 서술을 재생성한다.
#
# 제출: qsub -N G1C8_hsm_regenV2 -v PAYLOAD="scripts/regenV2_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }
for model in videollama2_av qwen2_5_omni_7b; do
  ids=logs/regen_ids_${model}.txt          # chain(B)이 만든 top-15
  ids_avcd=logs/regen_ids_avcd_${model}.txt # top-6
  [ -f "$ids" ] || { echo "skip: $ids 없음"; continue; }
  for m in base vcd_ext mad; do
    echo "regenV2: $model $m (free-form 192tok)"
    conda run -n $(env_for $model) --no-capture-output python -m src.runner \
      --model $model --method $m --benchmark avhbench \
      --ids-file $ids --max-new-tokens 192 --out-tag free192 \
      --set 'prompts.avhbench_suffix.'$model'=""' \
      --set monitoring.enabled=false > logs/regen2_${model}_${m}.log 2>&1 || true
  done
  echo "regenV2: $model avcd (free-form 96tok)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark avhbench \
    --ids-file $ids_avcd --max-new-tokens 96 --out-tag free96 \
    --set 'prompts.avhbench_suffix.'$model'=""' \
    --set monitoring.enabled=false > logs/regen2_${model}_avcd.log 2>&1 || true
done
echo '[regenV2] 출력 길이 확인 (med가 한 자릿수면 실패):'
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob('results/runs/avhbench/*__free*.jsonl')):
    lens = [len((json.loads(l).get('prediction') or '')) for l in open(f)]
    if lens:
        print(f, 'n=', len(lens), 'len min/med/max=',
              min(lens), sorted(lens)[len(lens)//2], max(lens))
EOF
echo REGEN2-DONE
