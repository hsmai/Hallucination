#!/usr/bin/env python3
"""아이디어 트랙 자동 판정 (무인 체인용) → logs/steer_judgment.json

① 재현 일치율: text_subtract 스티어링 예측 vs 선배 contam 예측 (per-sample)
② β 선택: Language Dom 100에서 β∈{0.5,1.0,2.0} 최고 평균 acc
③ go/no-go: best-β amplify가 같은 100샘플의 text_subtract보다 +1.0%p 이상 && 일치율 ≥85%
"""
import json, os, re, sys
from pathlib import Path

O = '/home3/t202401082/omni-steering/probe/outputs'
QA = '/home3/t202401082/omni-steering/Dataset/AVH_Bench/QA.json'
R = Path('results/steer/runs')

def norm(a):
    a = str(a).lower().strip(); a = re.sub(r'[^\w\s]$', '', a)
    if a in ['yes','y','true','1','correct']: return 'yes'
    if a in ['no','n','false','0','incorrect']: return 'no'
    if 'yes' in a: return 'yes'
    if 'no' in a: return 'no'
    return a

def load_jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def acc(recs):
    ok = [r for r in recs if not r['prediction'].startswith('ERROR:')]
    return (sum(norm(r['prediction']) == norm(r['ground_truth']) for r in ok) / len(ok) * 100
            if ok else None), len(ok)

# ---------- ① 재현 일치율 ----------
agree_n = agree_hit = 0
try:
    qa = json.load(open(QA))
    pos, cnt, seen = {}, {'Video-driven Audio Hallucination':0,'Audio-driven Video Hallucination':0}, set()
    for rec in qa:
        t = rec['task']
        if 'AV' in t: continue
        idx = cnt[t]; cnt[t] += 1
        k = (rec['video_id'], rec['text'])
        if k in seen: continue
        seen.add(k); pos[k] = (t, idx)
    for model, files in [('qwen2_5_omni_7b', ['avh_contam_va.json','avh_contam_av.json']),
                         ('videollama2_av', ['vl2_contam_avh_va_L22.json','vl2_contam_avh_av_L22.json'])]:
        theirs = {}
        for task, fn in zip(['Video-driven Audio Hallucination','Audio-driven Video Hallucination'], files):
            recs = [r for r in qa if r['task'] == task]
            res = json.load(open(os.path.join(O, fn)))['results']
            ptr = 0
            for i, rec in enumerate(recs):
                if ptr < len(res) and res[ptr]['sample_id'] == rec['video_id'] and res[ptr]['gt'] == norm(rec['label']):
                    theirs[(task, i)] = res[ptr]; ptr += 1
        for r in load_jl(R / 'avhbench' / f'{model}__ours_steer__repro.jsonl'):
            k = (r['video_id'], r['question'])
            if k not in pos or pos[k] not in theirs: continue
            agree_n += 1; agree_hit += (norm(r['prediction']) == theirs[pos[k]]['contam'])
    for model, fn in [('qwen2_5_omni_7b','cmm_contam.json'), ('videollama2_av','vl2_contam_cmm_L22.json')]:
        their = {x['sample_id']: x for x in json.load(open(os.path.join(O, fn)))['results']}
        for r in load_jl(R / 'cmm' / f'{model}__ours_steer__repro.jsonl'):
            t = their.get(r['video_id'])
            if not t: continue
            agree_n += 1; agree_hit += (norm(r['prediction']) == t['contam'])
except Exception as e:
    print(f'[judge] 재현 대조 오류(계속): {e}')
agreement = agree_hit / agree_n * 100 if agree_n else 0.0

# ---------- ② β 선택 ----------
betas = {}
for b in ('0.5', '1.0', '2.0'):
    accs = []
    for model in ('videollama2_av', 'qwen2_5_omni_7b'):
        a, n = acc(load_jl(R / 'cmm' / f'{model}__ours_steer__beta{b}.jsonl'))
        if a is not None: accs.append(a)
    if accs: betas[b] = sum(accs) / len(accs)
best_beta = max(betas, key=betas.get) if betas else None

# ---------- ③ text_subtract 대조 (같은 langdom 100) ----------
ts = []
for model in ('videollama2_av', 'qwen2_5_omni_7b'):
    a, n = acc(load_jl(R / 'cmm' / f'{model}__ours_steer__ts_langdom.jsonl'))
    if a is not None: ts.append(a)
ts_mean = sum(ts) / len(ts) if ts else None

go = bool(best_beta and ts_mean is not None and agreement >= 85.0
          and betas[best_beta] >= ts_mean + 1.0)
out = {'agreement_pct': round(agreement, 2), 'agreement_n': agree_n,
       'beta_accs': betas, 'best_beta': best_beta,
       'text_subtract_langdom': ts_mean, 'go': go}
Path('logs').mkdir(exist_ok=True)
json.dump(out, open('logs/steer_judgment.json', 'w'), indent=2)
print('[judge]', json.dumps(out, ensure_ascii=False))
