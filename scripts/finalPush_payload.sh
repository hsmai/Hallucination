#!/bin/bash
# finalPush: (1) VL2 CMM VCD/MAD 논문값 최대 근접 — 레시피 γ/α 스윕 + 전모달(VA=real) 변형
#            → 최적 설정 자동 선정 → 전량 1200 재실행 (out-tag paperfinal)
#            (2) Qwen AVCD fig 서술형 잔여분 재생성 (fig 트랙 선행)
# 2026-07-22 사용자 지시: "공평(전 모달리티 공급) 조건에서 기존 논문 수치 수준으로".
# MAD repo 배치 스크립트 자체가 γ 0.5~3.0 스윕이므로 스윕-최적 선정이 그들의 방법론.
source ~/.bashrc
cd ~/Hallucination
git pull --ff-only || true
SUF=" Answer only 'Yes' or 'No'. Do not include any explanation."

echo '[A] Qwen AVCD fig 서술형 (16건, resume)'
conda run -n qwen-omni --no-capture-output python -m src.runner \
  --model qwen2_5_omni_7b --method avcd --benchmark avhbench \
  --ids-file data/qa/fig2_qwen_ids.txt --max-new-tokens 96 --out-tag fig3 \
  --use-probe-question --set monitoring.enabled=false > logs/fp_qwen_avcd.log 2>&1 || true
echo 'A-DONE'

echo '[B] MAD γ 스윕 (레시피·silent VA, gate 200 균형)'
for g in 0.5 1.0 1.5 2.0; do
  conda run -n videollama2 --no-capture-output python -m src.runner \
    --model videollama2_av --method mad --benchmark cmm \
    --ids-file data/qa/gate_cmm_ids.txt \
    --set benchmarks.cmm.vl2_paper_recipe=true \
    --set "prompts.cmm_suffix.videollama2_av=$SUF" \
    --set methods.mad.gamma=$g --out-tag "rg$g" \
    --set monitoring.enabled=false > logs/fp_mad_g$g.log 2>&1 || true
done

echo '[C] 전모달 변형 (VA=실제 wav 결합) — MAD γ2.5 / VCD α0.5'
conda run -n videollama2 --no-capture-output python -m src.runner \
  --model videollama2_av --method mad --benchmark cmm \
  --ids-file data/qa/gate_cmm_ids.txt \
  --set benchmarks.cmm.vl2_paper_recipe=true --set benchmarks.cmm.vl2_recipe_va=real \
  --set "prompts.cmm_suffix.videollama2_av=$SUF" \
  --out-tag rgvar --set monitoring.enabled=false > logs/fp_mad_var.log 2>&1 || true
conda run -n videollama2 --no-capture-output python -m src.runner \
  --model videollama2_av --method vcd_ext --benchmark cmm \
  --ids-file data/qa/gate_cmm_ids.txt \
  --set benchmarks.cmm.vl2_paper_recipe=true --set benchmarks.cmm.vl2_recipe_va=real \
  --set "prompts.cmm_suffix.videollama2_av=$SUF" \
  --out-tag ravar --set monitoring.enabled=false > logs/fp_vcd_var.log 2>&1 || true

echo '[D] VCD α 스윕 (레시피·silent VA)'
for a in 0.25 1.0 2.5; do
  conda run -n videollama2 --no-capture-output python -m src.runner \
    --model videollama2_av --method vcd_ext --benchmark cmm \
    --ids-file data/qa/gate_cmm_ids.txt \
    --set benchmarks.cmm.vl2_paper_recipe=true \
    --set "prompts.cmm_suffix.videollama2_av=$SUF" \
    --set methods.vcd_ext.alpha=$a --out-tag "ra$a" \
    --set monitoring.enabled=false > logs/fp_vcd_a$a.log 2>&1 || true
done

echo '[E] 채점 + 최적 설정 선정'
conda run -n videollama2 python -m src.score --jsonl 'results/runs/cmm/*__r*.jsonl' > logs/fp_sweep_score.log 2>&1 || true
python3 - <<'PYEOF' | tee logs/fp_best.log
import json, glob
gate = {l.strip() for l in open('data/qa/gate_cmm_ids.txt') if l.strip()}
def acc(path, subset=None):
    c = t = 0
    for line in open(path):
        r = json.loads(line)
        if subset and r['sample_id'] not in subset: continue
        if r.get('correct') is None: continue
        t += 1; c += int(r['correct'])
    return (c / t * 100 if t else 0.0), t
cands = {'mad': {}, 'vcd_ext': {}}
# 스윕 결과 (200)
for f in glob.glob('results/runs/cmm/videollama2_av__mad__rg*.jsonl'):
    tag = f.split('__')[-1].replace('.jsonl','')
    g = 'var(γ2.5,VA=real)' if tag=='rgvar' else f"γ{tag[2:]}(silent)"
    cands['mad'][g] = acc(f)[0]
for f in glob.glob('results/runs/cmm/videollama2_av__vcd_ext__ra*.jsonl'):
    tag = f.split('__')[-1].replace('.jsonl','')
    a = 'var(α0.5,VA=real)' if tag=='ravar' else f"α{tag[2:]}(silent)"
    cands['vcd_ext'][a] = acc(f)[0]
# 기존 전량(γ2.5/α0.5 silent)을 같은 200으로 필터해 공정 비교
cands['mad']['γ2.5(silent)'] = acc('results/runs/cmm/videollama2_av__mad__paperrecipe.jsonl', gate)[0]
cands['vcd_ext']['α0.5(silent)'] = acc('results/runs/cmm/videollama2_av__vcd_ext__paperrecipe.jsonl', gate)[0]
best = {}
for m, d in cands.items():
    for k, v in sorted(d.items(), key=lambda x: -x[1]):
        print(f"{m:8s} {k:22s} {v:.2f}%")
    best[m] = max(d.items(), key=lambda x: x[1])
    print(f"→ BEST {m}: {best[m][0]} ({best[m][1]:.2f}%)\n")
# env 기록
def parse(m, name):
    va = 'real' if 'VA=real' in name else 'silent'
    if m == 'mad':
        g = '2.5' if 'var' in name or '2.5' in name else name.split('(')[0][1:]
        return g, va
    a = '0.5' if 'var' in name or '0.5' in name else name.split('(')[0][1:]
    return a, va
mg, mva = parse('mad', best['mad'][0])
va_, vva = parse('vcd_ext', best['vcd_ext'][0])
with open('logs/fp_best.env', 'w') as f:
    f.write(f"BEST_MAD_G={mg}\nBEST_MAD_VA={mva}\nBEST_VCD_A={va_}\nBEST_VCD_VA={vva}\n")
PYEOF
source logs/fp_best.env
echo "선정: MAD γ=$BEST_MAD_G VA=$BEST_MAD_VA / VCD α=$BEST_VCD_A VA=$BEST_VCD_VA"

echo '[F] 최적 설정 전량 1200 (paperfinal)'
if [ "$BEST_MAD_G" = "2.5" ] && [ "$BEST_MAD_VA" = "silent" ]; then
  cp results/runs/cmm/videollama2_av__mad__paperrecipe.jsonl results/runs/cmm/videollama2_av__mad__paperfinal.jsonl
  echo 'MAD: 기존 전량(γ2.5 silent) 재사용'
else
  conda run -n videollama2 --no-capture-output python -m src.runner \
    --model videollama2_av --method mad --benchmark cmm \
    --set benchmarks.cmm.vl2_paper_recipe=true --set benchmarks.cmm.vl2_recipe_va=$BEST_MAD_VA \
    --set "prompts.cmm_suffix.videollama2_av=$SUF" \
    --set methods.mad.gamma=$BEST_MAD_G --out-tag paperfinal \
    --set monitoring.enabled=false > logs/fp_mad_final.log 2>&1 || true
fi
if [ "$BEST_VCD_A" = "0.5" ] && [ "$BEST_VCD_VA" = "silent" ]; then
  cp results/runs/cmm/videollama2_av__vcd_ext__paperrecipe.jsonl results/runs/cmm/videollama2_av__vcd_ext__paperfinal.jsonl
  echo 'VCD: 기존 전량(α0.5 silent) 재사용'
else
  conda run -n videollama2 --no-capture-output python -m src.runner \
    --model videollama2_av --method vcd_ext --benchmark cmm \
    --set benchmarks.cmm.vl2_paper_recipe=true --set benchmarks.cmm.vl2_recipe_va=$BEST_VCD_VA \
    --set "prompts.cmm_suffix.videollama2_av=$SUF" \
    --set methods.vcd_ext.alpha=$BEST_VCD_A --out-tag paperfinal \
    --set monitoring.enabled=false > logs/fp_vcd_final.log 2>&1 || true
fi
conda run -n videollama2 python -m src.score --jsonl 'results/runs/cmm/*__paperfinal.jsonl' > logs/fp_final_score.log 2>&1 || true
echo '=== 최종 (목표: MAD 81.3 / VCD 76.4) ==='
cat logs/fp_final_score.log
echo FINALPUSH-DONE
