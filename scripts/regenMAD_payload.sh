#!/bin/bash
# regenMAD: Qwen MAD 서술형 raw text 교정 재생성 — γ=2.5→0.5 (정성 전용).
#
# 근거: MAD 논문 16p는 Qwen×MAD의 정상 서술형을 제시. 원인 추적 결과 repo의 AVHBench
# 평가 스크립트(eval_batch_mad.py) 기본 γ=0.5 (Qwen·VL2 동일; γ=2.5는 CMM 스크립트).
# 우리 free192v2는 γ=2.5로 192토큰 생성 → 대조 과격으로 반복 붕괴("WellWell...").
# 정량 표는 γ=2.5 유지(게이트 대조 통과 설정), 이 job은 정성 샘플 15건만 재생성.
#
# 제출: qsub -N G1C8_hsm_regenMAD -W depend=afterany:<fixCPU> -v PAYLOAD="scripts/regenMAD_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true

echo '[1/2] Qwen MAD 서술형 재생성 (γ=0.5, probe 프롬프트)'
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method mad --benchmark avhbench \
  --ids-file logs/regen_ids_qwen2_5_omni_7b.txt --max-new-tokens 192 --out-tag free192v3 \
  --use-probe-question --set methods.mad.gamma=0.5 --set monitoring.enabled=false \
  > logs/regenMAD_qwen.log 2>&1 || true

echo '  출력 길이/품질 확인:'
python3 - <<'EOF'
import json
f='results/runs/avhbench/qwen2_5_omni_7b__mad__free192v3.jsonl'
try:
    lens=[]
    for l in open(f):
        r=json.loads(l); p=r.get('prediction') or ''
        lens.append(len(p))
        toks=p.split()
        flag='DEGEN' if toks and max(toks.count(w) for w in set(toks))>len(toks)*0.5 and len(toks)>=20 else 'ok'
        print(' ', r['sample_id'][:45], f'len={len(p)}', flag)
    print('  med len =', sorted(lens)[len(lens)//2] if lens else 0)
except FileNotFoundError:
    print('  파일 없음 — 재생성 실패')
EOF

echo '[2/2] Qwen 정성 샘플 재패키징 (v3가 glob 정렬 마지막이라 MAD는 v3 우선 병합)'
conda run -n videollama2 --no-capture-output python scripts/package_samples.py \
  --mining results/runs/mining/avhbench__qwen2_5_omni_7b.csv \
  --media-dir /home3/t202401082/omni-steering/Dataset/AVH_Bench \
  --benchmark avhbench --top 15 \
  --model qwen2_5_omni_7b --regen-dir results/runs/avhbench \
  --out results/runs/qualitative/avhbench__qwen2_5_omni_7b \
  > logs/package3_qwen.log 2>&1 || true
echo REGENMAD-DONE
