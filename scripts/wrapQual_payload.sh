#!/bin/bash
# wrapQual: 정성 샘플 마무리 — AVH AVCD 확장(6→15) + AVH·CMM 전체 재패키징.
#
# 배경: cmmQual의 CMM AVCD 서술형이 샘플당 평균 27분(클립 무거움)으로 완주 시 +4~5h.
# 사용자 결정(2026-07-20): 8/15에서 절단 — 마이닝 최상위 8건이면 Figure 목적 충분.
# 이 payload는 cmmQual의 남은 두 단계([2] AVH AVCD 확장, [4] 재패키징)만 수행.
#
# 제출: qsub -N G1C8_hsm_wrapQual -W depend=afterany:<fixLast2> -v PAYLOAD="scripts/wrapQual_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/2] AVH AVCD 서술형 확장 (top-6 → top-15, free96v2 재개)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n $(env_for $model) --no-capture-output python -m src.runner \
    --model $model --method avcd --benchmark avhbench \
    --ids-file logs/regen_ids_${model}.txt --max-new-tokens 96 --out-tag free96v2 \
    --use-probe-question --set monitoring.enabled=false \
    > logs/wrap_avh_avcd_${model}.log 2>&1 || true
done

echo '[2/2] 패키징 — CMM 신규(AVCD 텍스트는 상위 8건) + AVH 재패키징(AVCD 15건 반영)'
for model in videollama2_av qwen2_5_omni_7b; do
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/cmm__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/CMM_dataset \
    --benchmark cmm --top 15 \
    --model ${model} --regen-dir results/runs/cmm \
    --out results/runs/qualitative/cmm__${model} \
    > logs/wrap_pkg_cmm_${model}.log 2>&1 || true
  conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
    --mining results/runs/mining/avhbench__${model}.csv \
    --media-dir /home3/t202401082/omni-steering/Dataset/AVH_Bench \
    --benchmark avhbench --top 15 \
    --model ${model} --regen-dir results/runs/avhbench \
    --out results/runs/qualitative/avhbench__${model} \
    > logs/wrap_pkg_avh_${model}.log 2>&1 || true
done
echo '폴더 수 확인:'
for d in results/runs/qualitative/*/; do echo "  $d: $(ls -d $d*/ 2>/dev/null | wc -l)개"; done
echo WRAPQUAL-DONE
