#!/bin/bash
# figCand: figure 전용 2차 후보 재생성 — 라벨 무결(캡션 검수) + 4방법 전부 오답(Y/N) +
# VdAH·GT=No("보이지만 소리 안 남") 유형만. 서술형에서 4방법 모두 환각을 보이는
# 샘플을 확보하기 위한 재생성 (기존 마이닝 top-15의 품질 실패 대응, 2026-07-20).
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }
for model in videollama2_av qwen2_5_omni_7b; do
  ids=data/qa/fig_ids_${model}.txt
  for m in base vcd_ext mad; do
    GAMMA_SET=""
    [ "$model" = qwen2_5_omni_7b ] && [ "$m" = mad ] && GAMMA_SET="--set methods.mad.gamma=0.5"
    echo "figCand: $model $m"
    conda run -n $(env_for $model) --no-capture-output python -m src.runner \
      --model $model --method $m --benchmark avhbench \
      --ids-file $ids --max-new-tokens 192 --out-tag fig192 \
      --use-probe-question $GAMMA_SET --set monitoring.enabled=false \
      > logs/figcand_${model}_${m}.log 2>&1 || true
  done
  echo "figCand: $model avcd"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark avhbench \
    --ids-file $ids --max-new-tokens 96 --out-tag fig96 \
    --use-probe-question --set monitoring.enabled=false \
    > logs/figcand_${model}_avcd.log 2>&1 || true
done
echo '[figCand] 출력 요약:'
python3 - <<'PYEOF'
import json, glob
for f in sorted(glob.glob('results/runs/avhbench/*__fig*.jsonl')):
    lens=[len((json.loads(l).get('prediction') or '')) for l in open(f)]
    if lens: print(' ', f.split('/')[-1], 'n=', len(lens), 'med=', sorted(lens)[len(lens)//2])
PYEOF
echo FIGCAND-DONE
