#!/bin/bash
# fixLast2: 최후 2건(wSalQi·S63S9z, ~28k 토큰) — q-block 청킹으로 해결 + 에러 0 최종 표.
#
# 근거: 마스킹 교정 후 남은 병목 = 헤드당 softmax fp32 [1,L,S] 3.02GB (traceback 실증).
# 교정: 초장문(>16384) prefill에서 query 행 블록(4096) 분할 — softmax/마스킹/재정규화/
# aw@v 전부 행 단위 연산이라 수치 동일 (로컬 함수 테스트 + 서버 통합 테스트로 검증).
# keep(마스킹 기준)은 마지막 query 행 선계산으로 공유 — AVCD 정의 그대로.
#
# 제출: qsub -N G1C8_hsm_fixLast2 -W depend=afterany:<cmmQual> -v PAYLOAD="scripts/fixLast2_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true

echo '[1/3] 서버 스모크: q-block 통합 테스트 (신규 구역 4단 검증)'
conda run -n qwen-omni --no-capture-output python -m pytest \
  tests/test_attn_patch_qwen452.py tests/test_avcd_attention.py -q \
  > logs/fixL2_pytest.log 2>&1
if [ $? -ne 0 ]; then
  echo '  ✗ 서버 테스트 실패 — 재실행 중단 (logs/fixL2_pytest.log 확인)'
  tail -5 logs/fixL2_pytest.log
  exit 1
fi
tail -1 logs/fixL2_pytest.log

echo '[2/3] 잔여 2건 재실행 (q-block 경로)'
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark cmm --retry-errors \
  > logs/fixL2_qwen_avcd_cmm.log 2>&1 || true

echo '[3/3] 최종 채점·집계 — 목표(1) 최종 표'
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
echo "FIXLAST2-DONE (잔여에러=$REMAIN)"
