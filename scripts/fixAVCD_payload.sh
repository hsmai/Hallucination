#!/bin/bash
# fixAVCD: AVCD 30건 최종 해결 — 마스킹 메모리 교정판으로 full GPU 재실행 + 에러 0 최종 표.
#
# 근거(diagOOM traceback): OOM 지점 = mask_attention_rows의 masked/denom —
# 시퀀스 ~20k에서 (H,L,S) fp32 복사본 3개(ones_like/곱/나눗셈)가 피크에 +4.6GB.
# 교정: 행-스케일 (H,L,1) 브로드캐스트 + in-place 곱/나눗셈 (수식 동일 —
# tests/test_avcd_attention.py::test_inplace_equivalence_and_purity에서 교정 전
# 수식과 allclose 검증). 오프로드·CPU 불필요, 동일 조건(GPU bf16) 유지.
#
# 제출: qsub -N G1C8_hsm_fixAVCD -v PAYLOAD="scripts/fixAVCD_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
env_for() { [ "$1" = qwen2_5_omni_7b ] && echo qwen-omni || echo videollama2; }

echo '[1/2] AVCD 잔여분 재실행 (마스킹 메모리 교정판, full GPU)'
# AVH 17건: fixCPU가 ERROR 기록을 제거한 채 중단 → 파일에 부재 = 러너가 자동으로 남음 처리
# CMM 13건: ERROR 기록 잔존 → --retry-errors로 제거 후 재시도
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark avhbench --retry-errors \
  > logs/fixA_qwen_avcd_avhbench.log 2>&1 || true
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark cmm --retry-errors \
  > logs/fixA_qwen_avcd_cmm.log 2>&1 || true

echo '[2/2] 최종 채점·집계·마이닝 — 목표(1) 최종 표'
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
echo "FIXAVCD-DONE (잔여에러=$REMAIN)"
