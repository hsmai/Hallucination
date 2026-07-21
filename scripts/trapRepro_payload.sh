#!/bin/bash
# trapRepro: VL2 VCD/MAD × CMM 논문 수치 격차의 원인 실증 — MAD repo식 입력 재현 run.
#
# 가설(기존 정황 3종 확보): MAD 논문의 CMM 수치는 별도 wav 대신 무음 mp4 트랙을
# audio branch에 공급한 입력 위에서 나온 것. 이 run은 benchmarks.cmm.trap_emulation=true로
# 그 입력을 재현해 논문 수치(MAD 81.3 / VCD 76.4)가 복원되는지 실증한다.
# 본 결과 무변경 (별도 __trap 태그 파일, 집계 제외).
#
# 제출: qsub -N G1C8_hsm_trapRepro -W depend=afterany:<wrapQual> -v PAYLOAD="scripts/trapRepro_payload.sh" scripts/pbs_job.sh
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true

for m in mad vcd_ext; do
  echo "trapRepro: videollama2_av $m cmm (MAD repo식 입력)"
  conda run -n videollama2 --no-capture-output python -m src.runner \
    --model videollama2_av --method $m --benchmark cmm \
    --set benchmarks.cmm.trap_emulation=true --out-tag trap \
    --set monitoring.enabled=false \
    > logs/trap_${m}.log 2>&1 || true
done

conda run -n videollama2 python -m src.score --jsonl 'results/runs/cmm/*__trap.jsonl' > logs/trap_score.log 2>&1 || true
echo '=== 함정 재현 결과 (논문: MAD CMM 81.3 / VCD CMM 76.4, 우리 교정입력: 70.9 / 58.5) ==='
conda run -n videollama2 python - <<'EOF'
import json, glob
from collections import defaultdict
CAT = {"overrely_visual_ignore_audio": "Visual", "overrely_audio_ignore_visual": "Audio",
       "overrely_language_ignore_visual": "Language"}
for f in sorted(glob.glob('results/runs/cmm/*__trap.jsonl')):
    tally = defaultdict(lambda: [0, 0])
    for line in open(f):
        r = json.loads(line)
        if r.get("correct") is None: continue
        t = tally[CAT.get(r["category"], "?")]
        t[1] += 1; t[0] += int(r["correct"])
    tot = [sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())]
    cols = "  ".join(f"{k} {v[0]/v[1]*100:.1f}" for k, v in sorted(tally.items()) if v[1])
    print(f"{f.split('/')[-1]}: {cols}  | Overall {tot[0]/max(tot[1],1)*100:.1f} (n={tot[1]})")
EOF
echo TRAP-REPRO-DONE
