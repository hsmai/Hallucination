#!/bin/bash
# cmmQual: CMM 기반 정성 샘플 준비 + AVH AVCD 서술형 15건 확장.
#
# 배경: MAD 논문 Fig.9-10의 yes/no 예시 문체("Did you see any sheep?")가 CMM 문항
# 템플릿과 일치 → 선배 대비 CMM 기반 정성 샘플도 준비 (2026-07-20 사용자 지시).
# + AVH AVCD 서술형이 top-6뿐이라 추천 샘플(00092__013 등)에 누락 → 15건 전체로 확장.
# Qwen MAD 서술형은 γ=0.5 (repo AVH 평가 기본값 — γ=2.5는 Qwen 장문 반복 붕괴 실증).
# 정량 결과 무변경: 전부 별도 태그(free*) 파일 + 패키징만 갱신.
#
# 제출: qsub -N G1C8_hsm_cmmQual -W depend=afterany:<fixAVCD> -v PAYLOAD="scripts/cmmQual_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/4] CMM 서술형 재생성 (probe, top-15)'
for model in videollama2_av qwen2_5_omni_7b; do
  ids_src=results/runs/mining/cmm__${model}.ids.txt
  [ -f "$ids_src" ] || { echo "skip: $ids_src 없음"; continue; }
  head -15 "$ids_src" > logs/cmm_regen_ids_${model}.txt
  for m in base vcd_ext mad; do
    GAMMA_SET=""
    [ "$model" = qwen2_5_omni_7b ] && [ "$m" = mad ] && GAMMA_SET="--set methods.mad.gamma=0.5"
    echo "cmmQual: $model $m (probe 192tok) $GAMMA_SET"
    conda run -n $(env_for $model) --no-capture-output python -m src.runner \
      --model $model --method $m --benchmark cmm \
      --ids-file logs/cmm_regen_ids_${model}.txt --max-new-tokens 192 --out-tag free192 \
      --use-probe-question $GAMMA_SET --set monitoring.enabled=false \
      > logs/cmmqual_${model}_${m}.log 2>&1 || true
  done
  echo "cmmQual: $model avcd (probe 96tok, 15건)"
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark cmm \
    --ids-file logs/cmm_regen_ids_${model}.txt --max-new-tokens 96 --out-tag free96 \
    --use-probe-question --set monitoring.enabled=false \
    > logs/cmmqual_${model}_avcd.log 2>&1 || true
done

echo '[2/4] AVH AVCD 서술형 확장 (top-6 → top-15, 같은 태그 free96v2 재개)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark avhbench \
    --ids-file logs/regen_ids_${model}.txt --max-new-tokens 96 --out-tag free96v2 \
    --use-probe-question --set monitoring.enabled=false \
    > logs/cmmqual_avh_avcd_${model}.log 2>&1 || true
done

echo '[3/4] 출력 품질 확인:'
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob('results/runs/cmm/*__free*.jsonl') + glob.glob('results/runs/avhbench/*__free96v2.jsonl')):
    lens=[len((json.loads(l).get('prediction') or '')) for l in open(f)]
    if lens:
        print(' ', f.split('/')[-1], 'n=', len(lens), 'len med=', sorted(lens)[len(lens)//2])
EOF

echo '[4/4] 패키징 — CMM 신규 + AVH 재패키징(AVCD 확장 반영)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/cmm__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/CMM_dataset \
    --benchmark cmm --top 15 \
    --model ${model} --regen-dir results/runs/cmm \
    --out results/runs/qualitative/cmm__${model} \
    > logs/package_cmm_${model}.log 2>&1 || true
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/avhbench__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/AVH_Bench \
    --benchmark avhbench --top 15 \
    --model ${model} --regen-dir results/runs/avhbench \
    --out results/runs/qualitative/avhbench__${model} \
    > logs/package_avh2_${model}.log 2>&1 || true
done
echo CMMQUAL-DONE
