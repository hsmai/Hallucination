# Claude Code 작업 지시서

> 사용법: 이 파일과 blueprint.md를 repo(github.com/hsmai/Hallucination) 루트에 커밋한 뒤,
> 로컬에서 `cd Hallucination && claude` 실행 후 아래 "첫 메시지"를 붙여넣는다.

---

## 첫 메시지

```
이 repo는 Omni-Steering 논문의 비교군(Base/VCD-ext/AVCD/MAD) 재현 실험 작업 공간이다.
루트의 claude_code_master_prompt.md 전체를 읽고 지시대로 수행하라.
시작 전에 그 문서의 "제약"과 "작업 원칙" 섹션을 발췌해 CLAUDE.md로 저장·커밋한 뒤, Phase L1부터 진행하라.
```

---

## 프로젝트 컨텍스트

나는 Omni-Steering(omni-modal LLM의 cross-modal hallucination을 완화하는 training-free 기법, 선배 연구) 논문의 **비교군 실험** 담당이다. 기존 방법들을 동일 세팅으로 실행하여 정량 수치와 샘플별 출력을 확보한다. 전체 계획은 repo의 `blueprint.md`에 있으며, 본 문서와 충돌 시 blueprint가 우선한다.

실행 매트릭스 (총 16 run — 실행 자체는 서버에서, 지금은 로컬 준비 단계):
```
모델 2종:    VideoLLaMA2-AV, Qwen2.5-Omni-7B
방법 4종:    Base, VCD-extended, AVCD, MAD   (전부 training-free decoding)
벤치마크 2종: CMM(Visual/Audio/Language Dominance), AVHBench(Video-Driven/Audio-Driven Hallucination)
```

내 per-sample 출력은 선배의 OURS 결과와 **video_id+question 키로 join**되어 논문 Figure 샘플 선별에 쓰인다. 공통 키 유지가 필수다.

### 구현할 방법 요약
- **VCD-extended**: 4-forward(clean/video왜곡/audio왜곡/둘다왜곡) 후 `(1+3α)·logit_vaq − α·logit_ṽaq − α·logit_vãq − α·logit_ṽãq`
- **MAD** (공식 코드: third_party/MAD): 생성 전 1회 modality query prompt("To answer this question, which modality is needed (audio, video, or both)?") forward → 'video'/'audio'/'both' 토큰 logit softmax = (w_v, w_a, w_av) → 4개 contrastive branch를 γ·w로 가중합. γ=2.5, greedy. weight는 질문당 1회 고정
- **AVCD** (공식 코드: third_party/AVCD): 마지막 query 토큰 attention(layer 평균)으로 dominant modality 판정 → 덜 dominant modality의 attention 상위 P%(50%) 토큰 zeroing(마지막 layer 제외) → trimodal CD 결합식 → entropy-guided adaptive decoding(τ=0.6, 저엔트로피 스텝 skip) → adaptive plausibility constraint(β=0.1). **full attention weight 필요 → 전 방법 eager attention으로 통일**

## 제약 (CLAUDE.md로 발췌)

1. **로컬 = macOS(Apple Silicon), GPU 없음.** 7B 모델 로드/forward 금지. CUDA 전용 패키지(flash-attn 등) 설치 시도 금지. 개발·검증은 CPU + 소형 더미 텐서 + MockModel dry-run으로만 한다.
2. **서버 계정 발급 대기 중.** 서버(A100 80GB×1)에는 이미 다음이 존재한다:
   - conda 환경 2종: `/home3/t202401082/.conda/envs/qwen-omni`, `/home3/t202401082/.conda/envs/videollama2` → 발급 후 복제해서 사용
   - OURS 코드+데이터셋: `/home3/t202401082/omni-steering` → 발급 후 직접 참조
   따라서 **로컬에서 conda env 구축, 모델 가중치 다운로드, 미디어 데이터 다운로드를 하지 않는다.** 서버 작업은 runbook 문서로만 준비한다. 벤치마크 QA json(경량)만 로컬로 받는다.
3. **세팅 미확정 항목 존재.** 프롬프트 템플릿, AVHBench split, CMM 구성, 채점기는 서버의 OURS 코드에서 확정된다. **어떤 세팅도 하드코딩하지 말고 전부 `configs/unified_settings.yaml`에서 읽는다.** 미확정 값은 `UNKNOWN_pending_server` + 임시 기본값(MAD repo 세팅)으로 둔다.
4. 모든 산출물을 커밋한다(서버에서 git clone 하나로 이전). 대용량은 .gitignore.

## 작업 원칙 (CLAUDE.md로 발췌)

- Phase 완료마다 git commit(무엇을 검증했는지 메시지에 명시)하고, 나에게 (a)산출물 목록 (b)발견한 이슈 (c)다음 Phase 진행 여부를 보고한 뒤 진행한다.
- 논문 수치·하이퍼파라미터를 기억으로 단정하지 않는다. 본 문서의 요약과 clone한 실제 코드가 근거이며, 둘이 충돌하면 코드를 우선하되 반드시 보고한다.
- 외부 repo 코드는 수정하지 않고 `third_party/`에 원본 유지, 우리 코드는 `src/`에 작성한다.
- 스크립트는 재실행 안전(idempotent)하게, 실패 시 명확한 에러 메시지를 낸다.

## Phase L1 — 셋업 + 코드 정독

1. 디렉토리 구조 생성: `configs/ src/ src/methods/ scripts/ third_party/ data/qa/ results/ docs/ tests/`
2. `third_party/`에 clone:
   - https://github.com/top-yun/MAD
   - https://github.com/kaistmm/AVCD
   - https://github.com/DAMO-NLP-SG/VideoLLaMA2 의 **audio_visual branch**
3. 세 repo를 정독하고 `docs/code_analysis.md` 작성. 반드시 답할 것:
   - MAD: 디코딩 구현 파일/함수 위치, modality query prompt 문자열, weight 추출 코드, CMM/AVHBench 데이터 로더와 프롬프트 템플릿, score.py·score_cmm.py의 채점 규칙, **VCD-extended가 이미 구현되어 있는지**, qwen-omni/와 VideoLLaMA2/ 디렉토리의 구조 차이
   - AVCD: dominance 계산 위치, **modality 토큰 span(video/audio/text 인덱스 경계) 계산 방식**(Qwen 포팅의 핵심 재료), attentive masking·EAD·plausibility 구현 위치, AVHBench split 정의
   - VideoLLaMA2-AV: 모델 로드 방식, eager attention 강제 방법, audio/video 토큰이 시퀀스에 배치되는 구조
   - MAD와 AVCD 간 프롬프트/split/채점의 **불일치 사항 목록**
4. `configs/unified_settings.yaml` 초안: 확정값(γ=2.5, AVCD α·P·τ·β, greedy temp=0, seed=42, bf16, eager attention)과 UNKNOWN_pending_server 항목(프롬프트, split, CMM 구성, 채점기, 서버 데이터 경로 프리픽스)을 구분 표기

## Phase L2 — QA 데이터 + 로더

1. AVHBench(github.com/kaist-ami/AVHBench)와 CMM 공개 소스에서 **QA json/annotation 파일만** `data/qa/`로 다운로드(미디어 제외). 다운로드 불가 항목은 URL과 사유를 `docs/data_report.md`에 기록
2. `src/data.py` 공통 로더: (sample_id, video_path, audio_path, question, ground_truth, category)를 내놓는 벤치마크 공통 인터페이스. 미디어 경로는 yaml의 프리픽스로 조립하고, 로컬에서는 파일 존재검사를 skip하는 플래그 제공. 서버의 omni-steering 내부 QA 파일로 교체할 수 있도록 json 경로만 바꾸면 되는 구조로
3. 샘플 수·카테고리 분포를 `docs/data_report.md`에 기록

## Phase L3 — 하네스 (dry-run 완결)

1. `src/runner.py` — 방법 플러그인 구조:
   ```python
   class DecodingMethod:  # base / vcd_ext / avcd / mad 가 구현
       def setup(self, model, cfg): ...
       def generate(self, sample) -> dict:  # prediction + internals 반환
   ```
2. per-sample JSONL 로깅(blueprint의 D2 스키마 그대로) + 처리된 sample_id skip(중단-재개)
3. `--dry-run` 모드: 모델 로드/forward를 MockModel(고정 또는 랜덤 yes/no 반환)로 대체하여 **실제 QA json으로 16개 조합 전부 end-to-end** 통과 검증
4. `src/score.py`: MAD repo의 채점 로직 이식(임시 채점기). 추후 OURS 채점기로 교체 가능하도록 함수 단위 분리
5. `src/aggregate.py`: JSONL → 정량 테이블(md/csv) + 카테고리 세부표 자동 생성
6. dry-run → 채점 → 집계 전 구간 통과 후 commit

## Phase L4 — 방법 구현 + CPU 단위 테스트

1. `src/methods/mad.py`: MAD repo에서 이식 (VideoLLaMA2용/Qwen용 각각 — repo에 둘 다 존재)
2. `src/methods/vcd_ext.py`: MAD repo에 구현이 있으면 이식, 없으면 위 수식으로 신규 작성
3. `src/methods/avcd.py`: AVCD repo에서 VideoLLaMA2용 이식 + **Qwen2.5-Omni용 신규 포팅**(공식 코드 없음). 포팅 순서: Qwen processor 출력에서 modality 토큰 span 인덱스 파악 → attention weight 접근(output_attentions) → 마스킹 적용 지점
4. `tests/` — 소형 더미 텐서(hidden 64, seq 50, vocab 100)로 CPU 검증:
   - MAD: weight softmax → 4-branch 가중합 수식 일치
   - AVCD: dominance argmax, top-50% zeroing 마스크의 shape/개수, 결합식 계수, EAD 분기, plausibility 마스킹
   - VCD-ext: 계수 검증
   - 러너: 중단-재개, 로깅 스키마 필드 완전성
5. pytest 전체 통과 후 commit

## Phase L5 — 서버 runbook

`docs/server_runbook.md` 작성 — 서버 계정 발급 당일 위에서부터 복붙 순서대로 실행 가능하게:
1. 작업 repo clone
2. conda env 복제: `conda create -n qwen-omni --clone /home3/t202401082/.conda/envs/qwen-omni` (videollama2 동일). 실패 대비 fallback: `conda list -p <경로> --export`로 스펙 추출 후 신규 생성. 복제 후 `conda list` 스냅샷을 docs/에 커밋하는 명령 포함
3. **OURS 세팅 추출 체크리스트** — `/home3/t202401082/omni-steering`를 정독하여:
   - 인퍼런스 스크립트의 프롬프트 문자열과 디코딩 파라미터(generate 인자)
   - 데이터 로더의 AVHBench split·CMM 카테고리/샘플 수
   - 채점 스크립트 위치 → `src/score.py`로 이식
   - 실행 커맨드(쉘 스크립트/README/히스토리)
   - 데이터셋 실경로 → yaml 프리픽스 기록
   - OURS 및 Ours(Base)의 결과 JSON 존재 여부(있으면 경로 기록)
   - 각 항목으로 yaml의 UNKNOWN을 확정하는 절차 포함
4. 모델 가중치 위치 확인(HF 캐시/omni-steering 내), 없을 때만 다운로드 명령
5. 실행 스크립트 3종 준비:
   - `scripts/run_smoke.sh`: 16조합 각 5샘플
   - `scripts/run_gate.sh`: 방법별 100~200샘플 + 논문 수치 대조표 출력 + **Base가 선배 보고치(CMM 73.6/73.1, AVHBench 77.4/76.8)와 일치하는지 검증** + AVCD CMM용 α 그리드(0.5~3.0, 0.5 간격, 100샘플)
   - `scripts/run_full.sh`: 16 run을 모델 단위로 묶어 순차, nohup+로그, 우선순위(AVHBench 전 방법 → CMM MAD/AVCD → CMM VCD-ext)

## 진행 방식

- 지금 Phase L1부터 시작하라. Phase 하나가 끝날 때마다 보고 후 진행 여부를 확인받아라.
- 서버 계정이 발급되면 내가 알린다. 그 전까지 서버 의존 작업(env 활성화, 다운로드, 실제 모델 forward)은 시도하지 말고 runbook 문서화로만 준비하라.
