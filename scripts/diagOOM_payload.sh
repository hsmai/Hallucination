#!/bin/bash
# diagOOM: AVCD 30건 실패 원인 정밀 진단 — 대표 2건을 full GPU로 재실행해
# runner의 traceback 로깅(신규)으로 OOM 발생 연산의 정확한 위치를 특정한다.
#
# 배경: 길이(9~13s)·해상도(640x480, 통과군에도 흔함) 가설 모두 기각.
# 오프로드 시 49.36GiB 단일 할당 시도 관측 → 청킹이 안 걸린 경로 존재 의심.
# 본 job은 산출물 무변경(--out-tag diag, 별도 파일)이며 순수 진단용.
#
# 제출: qsub -N G1C8_hsm_diagOOM -W depend=afterany:<regenMAD> -v PAYLOAD="scripts/diagOOM_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true

printf '%s\n' 'FSKh7W::Did you hear sheep bleating in the audio?' > logs/diag_cmm_ids.txt
printf '%s\n' '01023::Is the plant making sound in the audio?' > logs/diag_avh_ids.txt

echo '[diag] CMM 대표 1건 (FSKh7W)'
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark cmm \
  --ids-file logs/diag_cmm_ids.txt --out-tag diag \
  --set monitoring.enabled=false > logs/diag_cmm.log 2>&1 || true

echo '[diag] AVH 대표 1건 (01023)'
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark avhbench \
  --ids-file logs/diag_avh_ids.txt --out-tag diag \
  --set monitoring.enabled=false > logs/diag_avh.log 2>&1 || true

echo '[diag] traceback 요약:'
for f in logs/diag_cmm.log logs/diag_avh.log; do
  echo "== $f =="
  grep -A 40 'traceback:' $f | grep -E 'File "|line [0-9]+, in|Error' | tail -15
  grep -E '완료:|CUDA out of memory. Tried to allocate' $f | tail -2
done
echo DIAG-DONE
