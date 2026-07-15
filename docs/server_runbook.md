# 서버 Runbook — 계정 발급 당일 위에서부터 순서대로 실행

> 대상: A100 80GB / RTX A6000 48GB / RTX 3090 Ti 24GB (§S1-2에서 분기).
> 전제: `/home3/t202401082/.conda/envs/{qwen-omni,videollama2}` 와 `/home3/t202401082/omni-steering` 읽기 권한.
> 권한 없으면 선배에게 chmod/그룹 요청 후 진행 (blueprint §9-1).
> 표기: ☐ = 확인 후 체크. 명령은 그대로 복붙 가능하도록 작성.

---

## S1-1. repo + 참조 코드 + 의존성

```bash
cd ~ && git clone https://github.com/hsmai/Hallucination && cd Hallucination
bash scripts/setup_third_party.sh        # MAD/AVCD/VideoLLaMA2 고정 커밋 클론
mkdir -p logs
```

## S1-2. GPU 확인 및 기종별 분기

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee logs/gpu_info.txt
```

| 기종 | 판정 | 조치 |
|---|---|---|
| A100 80GB / A6000 48GB | ☐ 그대로 진행 | 없음 (A6000은 MAD 논문과 동일 계열 — 속도만 A100 대비 ~2배) |
| **3090 Ti 24GB** | ☐ Qwen OOM 주의 | 스모크에서 Qwen 조합 OOM 시 → §트러블슈팅 T1 (offload). VideoLLaMA2는 영향 없음 |

## S1-3. conda env 복제 (+ 스냅샷 커밋)

```bash
conda create -y -n qwen-omni    --clone /home3/t202401082/.conda/envs/qwen-omni
conda create -y -n videollama2  --clone /home3/t202401082/.conda/envs/videollama2
```

☐ 복제 실패(권한/절대경로 문제) 시 fallback:

```bash
conda list -p /home3/t202401082/.conda/envs/qwen-omni   --export > docs/env_qwen_spec.txt
conda list -p /home3/t202401082/.conda/envs/videollama2 --export > docs/env_vl_spec.txt
conda create -y -n qwen-omni   --file docs/env_qwen_spec.txt
conda create -y -n videollama2 --file docs/env_vl_spec.txt
# pip 패키지는 export에 안 잡힘: /home3/.../envs/*/lib/python*/site-packages 대조 후 pip install
```

스냅샷 기록 (성공 경로든 fallback이든):

```bash
conda run -n qwen-omni   pip list > docs/env_qwen_snapshot.txt
conda run -n videollama2 pip list > docs/env_vl_snapshot.txt
conda run -n qwen-omni   pip install -r requirements.txt   # pyyaml/tqdm/pytest만 추가
conda run -n videollama2 pip install -r requirements.txt
git add docs/env_*.txt && git commit -m "서버 env 스냅샷"
```

☐ **transformers 버전 기록** (신규 구역 호환성의 1차 관문):

```bash
conda run -n qwen-omni   python -c "import transformers; print('qwen env:', transformers.__version__)"
conda run -n videollama2 python -c "import transformers; print('vl env:', transformers.__version__)"
# 로컬 검증 기준: 4.57.6 (qwen 패치), 4.42.3 (VideoLLaMA2 요구 버전)
```

## S1-4. OURS 세팅 추출 — yaml UNKNOWN 14개 확정 체크리스트

`/home3/t202401082/omni-steering` 를 정독하며 아래 표를 채운다.
**모든 항목은 configs/unified_settings.yaml 의 해당 키에 기록하고, 근거 파일 경로를 주석으로 남긴다.**

```bash
ls -R /home3/t202401082/omni-steering | head -100   # 구조 파악부터
grep -rn "Answer\|prompt" /home3/t202401082/omni-steering --include="*.py" -l  # 프롬프트 후보
grep -rn "generate\|temperature\|max_new" /home3/t202401082/omni-steering --include="*.py" -l
```

| ☐ | 확인할 것 | 어디서 (예상) | yaml 키 |
|---|---|---|---|
| ☐ | AVHBench 프롬프트 suffix | 인퍼런스 스크립트의 질문 조립부 | `prompts.avhbench_suffix` |
| ☐ | CMM 프롬프트 suffix | 〃 | `prompts.cmm_suffix` |
| ☐ | generate 인자 (max_new_tokens, temperature 등) | model.generate 호출부 | `decoding.max_new_tokens.*` |
| ☐ | AVHBench split (QA.json 전체-AV제외? val/test?) | 데이터 로더 | `benchmarks.avhbench.split` |
| ☐ | CMM 카테고리/샘플 수 (over-reliance 1,200인지) | 〃 | `benchmarks.cmm.categories` |
| ☐ | OURS QA json 실경로 (우리 로컬본과 diff로 동일성 확인) | 데이터 디렉토리 | `benchmarks.*.qa_json` (다르면 교체) |
| ☐ | 채점 스크립트 → 규칙이 MAD식과 같은가 | score/eval 스크립트 | `scoring.scorer` (§S1-4b) |
| ☐ | 미디어 실경로 프리픽스 | 로더의 경로 조립부 | `paths.avhbench_media_dir`, `paths.cmm_media_dir`, `paths.data_prefix` |
| ☐ | 모델 가중치 경로 | 로드부 or HF 캐시 | `models.*.local_path` |
| ☐ | attention 구현/dtype (eager·bf16 확인) | 모델 로드부 | (불일치 시 보고 후 결정) |
| ☐ | **OURS·Ours(Base) per-sample 결과 JSON 존재 여부** | results/ 유사 디렉토리 | 있으면 경로 기록 → S4 join에 사용, 선배에게 재요청 불필요 |
| ☐ | 실행 커맨드 기록 (sh/README/bash history) | repo 루트 | 참고용 docs/에 복사 |

### S1-4b. 채점기 이식

```bash
# OURS 채점 스크립트를 찾아 규칙 확인. MAD식(yes우선 정규식)과 동일하면 그대로,
# 다르면 src/score.py에 함수 추가:
#   1) SCORERS["ours"] = ours_is_correct 구현 (원 스크립트 로직 그대로 이식)
#   2) yaml scoring.scorer: ours 로 변경
#   3) pytest tests/test_runner_pipeline.py::TestScorerRules 에 케이스 추가 후 실행
# 채점 스크립트가 없으면: MAD식 유지가 임시 결론임을 선배에게 보고 (blueprint §9-3)
```

확정 후:

```bash
python -c "from src.config import load_config; c=load_config(); print('남은 pending:', c.pending)"
# → 빈 리스트가 목표. 남은 항목은 사유를 커밋 메시지에 기록
git add configs/unified_settings.yaml && git commit -m "서버 세팅 확정: OURS 코드 대조 (근거 경로 주석 참조)"
```

## S1-5. 모델 가중치 확인

```bash
ls ~/.cache/huggingface/hub 2>/dev/null | grep -i -E "videollama|qwen" || true
find /home3/t202401082 -maxdepth 4 -iname "*VideoLLaMA2*" -o -iname "*Qwen2.5-Omni*" 2>/dev/null | head
# 있으면 → yaml models.*.local_path에 기록
# 없을 때만 다운로드:
#   conda run -n videollama2 huggingface-cli download DAMO-NLP-SG/VideoLLaMA2.1-7B-AV
#   conda run -n qwen-omni  huggingface-cli download Qwen/Qwen2.5-Omni-7B
```

## S1-6. 신규 구역 사전 점검 (CLAUDE.md 민감 구역 — 스모크 전에 반드시)

```bash
# 1) CPU 테스트 서버 재실행 (양쪽 env에서 — transformers 버전 차이 조기 발견)
conda run -n qwen-omni   python -m pytest tests/ -q
conda run -n videollama2 python -m pytest tests/ -q -k "not attn_patch"  # 4.42에는 eager_attention_forward 없음(정상)
```

☐ qwen env에서 `test_attn_patch_integration` 통과 — 실패 시: 서버 transformers 소스에서
   `eager_attention_forward` 위치 확인 후 `src/adapters/attn_patch.py`의 import 경로 수정

```bash
# 2) Qwen AVCD 실모델 최소 검증 (GPU, 샘플 1개 — span 좌표 정합 확인)
conda run -n qwen-omni python -m src.runner --model qwen2_5_omni_7b --method avcd \
  --benchmark avhbench --limit 1 --set paths.results_dir=results/precheck
```

☐ 에러 없이 1샘플 완료 + internals에 dominant/ead_skipped_ratio 기록됨
☐ "span 길이 != attention S" 에러 시 → §트러블슈팅 T3 (placeholder 확장 재매핑)
☐ ffmpeg 존재 확인: `which ffmpeg || conda install -y -n videollama2 ffmpeg`

## S1-7. 스모크 (16조합 × 5샘플)

```bash
bash scripts/run_smoke.sh 5 2>&1 | tee logs/smoke.log
```

☐ 16조합 전부 OK
☐ 각 로그 말미 VRAM 사용량 기록 (3090 Ti면 Qwen 조합 주시)
☐ 샘플당 처리 시간으로 소요시간표 계산:
   `전체 시간 ≈ Σ (방법별 s/sample × 샘플수)` — AVHBench 3,419 / CMM 1,200 기준.
   결과를 `docs/time_estimate.md`로 커밋하고 S3 야간 배치 계획 수립

---

## S2. 정합성 게이트

```bash
bash scripts/run_gate.sh 200 2>&1 | tee logs/gate.log
```

통과 기준 (logs/gate_table.log):

| ☐ | 검증 | 기준 | 실패 시 |
|---|---|---|---|
| ☐ | **Base ≈ Ours(Base)** | AVH 77.4/76.8, CMM 73.6~74.0/73.1 (±표본오차) | **여기서 멈춤.** 프롬프트→split→채점기→attention/dtype 순으로 yaml 항목별 diff 추적. 통과 전 다음 단계 금지 |
| ☐ | MAD 논문 Table 1 재현 | ±2%p (200샘플 노이즈 감안해 방향성 판단) | 해당 방법 internals 검토 (weight 분포/skip 비율) |
| ☐ | AVCD 행 | 참고 기준 (저자 재구현이라 정확 일치 미보장) | 어긋나면 수치와 원인 분석을 기록하고 진행 |
| ☐ | AVHBench split 판정 | AVCD val 81.95 재현 여부(β 판정과 동시) | split 가설 재검토 (avcd_test 72.15와 대조) |
| ☐ | α 그리드 (CMM) | 최고 acc α 확정 → yaml 기록 | — |
| ☐ | β 판정 | 81.95에 가까운 모드 → yaml faithful_mode 확정 | — |
| ☐ | VCD-ext α | Table 1 vcd_ext 행과 대조해 {0.5, 2.5} 중 결정 → yaml 기록 | 둘 다 어긋나면 1.0도 시도, 결과 기록 |

```bash
git add configs logs/gate*.log && git commit -m "S2 게이트: 판정 결과와 확정 하이퍼파라미터"
```

## S3. 본 실행 (야간 무인)

```bash
nohup bash scripts/run_full.sh > logs/full_$(date +%m%d_%H%M).log 2>&1 &
tail -f logs/full_*.log                 # 진행 확인
# 죽으면 같은 명령 재실행 — 처리된 sample_id는 자동 skip (중단-재개)
# GPU 시간 부족 시: P3(CMM vcd_ext) → P2 순으로 포기 (스크립트가 그 순서로 실행)
```

☐ 완료 후 `logs/full_final_table.log` = D1 테이블 (비볼드 수치 완성본)

## S4. 정리·납품

```bash
# 1) 최종 채점·집계·게이트 대조 확인
conda run -n videollama2 python -m src.aggregate --results results/runs

# 2) D3: 마이닝 (run_full.sh가 이미 실행) → 프레임 패키징
python scripts/package_samples.py \
  --mining results/runs/mining/avhbench__videollama2_av.csv \
  --media-dir <yaml의 avhbench_media_dir> \
  --out results/runs/qualitative/avhbench__videollama2_av --top 30
# (4개 조합 반복. CMM은 --benchmark cmm --media-dir <cmm_media_dir>)

# 3) 서술형 답변 재생성 (그림용 후보만, 방법 4종 × 후보):
for m in base vcd_ext mad avcd; do
  conda run -n videollama2 python -m src.runner --model videollama2_av --method $m \
    --benchmark avhbench --ids-file results/runs/mining/avhbench__videollama2_av.ids.txt \
    --max-new-tokens 256 --out-tag regen256
done

# 4) OURS join: S1-4에서 찾은 OURS per-sample 결과와 sample_id로 대조
#    (없으면 마이닝 csv 상태로 선배에게 전달 — OURS 정답 여부만 확인하면 Figure 후보 완성)

# 5) 납품물 커밋 (대용량 JSONL 원본은 서버 보관, 산출물만):
git add results/runs/tables results/runs/mining docs/time_estimate.md logs/full_final_table.log
git commit -m "S3/S4: D1 테이블 + D3 마이닝 목록 (최종)" && git push
```

납품 체크: ☐ D1 테이블 ☐ D2 JSONL(서버 보관, 경로 공유) ☐ D3 마이닝+패키징 ☐ D4 manifest(*.meta.json)·latency(JSONL의 inference_time_s 집계)·env 스냅샷

---

## 트러블슈팅

**T1. Qwen OOM (3090 Ti 24GB)** — 상황 확인 후 **한 가지 방향만** 구현한다 (사전 대응책 다중 구현 금지 — 2026-07-15 사용자 지시):
① `nvidia-smi`로 로드 직후/추론 중 사용량 확인 ② 짧은 비디오 샘플로 재시도해 경계 확인
③ 1순위 수정: **thinker-only 로드** (`Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained`로 교체
   — talker 가중치를 GPU에 안 올려 ~22GB→~17GB, 텍스트 경로 동일이라 수치 무영향.
   `src/adapters/qwen_omni.py` 로드부 + base의 generate 호출부 수정, 대응 테스트 갱신)
④ 그래도 부족하면: `device_map="auto", max_memory={0:"22GiB","cpu":"64GiB"}` CPU offload (수치 동일, 속도 저하)
⑤ 양자화는 금지 (비교 공정성)

**T2. conda 복제 실패** → S1-3 fallback (export → 신규 생성). pip 전용 패키지 누락 주의
(특히 qwen env의 `qwen-omni-utils`, vl env의 videollama2 의존성).

**T3. Qwen AVCD "span 길이 != attention S"**
placeholder가 임베딩 확장 후 1:1 유지되지 않는 경우. thinker 입력 좌표로 재매핑 필요:
`model.thinker.get_input_embeddings()` 직전의 실제 시퀀스에서 media 위치를 확인하고
`src/adapters/qwen_omni.py::_spans_for`를 확장 후 좌표 기준으로 수정 → 대응 테스트 추가
(CLAUDE.md 신규 구역 규칙: 수정 시 테스트 갱신 필수).

**T4. 패치 assert (eager_attention_forward 없음)**
서버 transformers 버전의 `modeling_qwen2_5_omni.py`를 열어 attention 호출부 확인 →
`install_patch()`에 넘길 모듈/함수명만 교체. 로컬 4.57.6 기준 검증 로직은 재사용.

**T5. videollama2 패키지 충돌 (fork 3종 동일 패키지명)**
증상: "videollama2 패키지가 이미 다른 백엔드에서 로드됨". 원인: 한 프로세스에서 두 백엔드.
해결: run당 새 프로세스 유지 (스크립트 기본). 커스텀 실행 시에도 method별 프로세스 분리.

**T6. OURS에 채점 스크립트 부재** → MAD식 채점 유지 + 선배 보고 (blueprint §9-3).
Ours(Base) 게이트가 통과하면 채점 방식이 사실상 동일하다는 방증이므로 그 결과를 함께 보고.
