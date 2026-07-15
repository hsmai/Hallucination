# Blueprint: Omni-Steering 비교군(Baseline) 실험

## 1. 프로젝트 개요

Omni-Steering은 omni-modal LLM(비디오+오디오+텍스트)의 cross-modal hallucination을 완화하는 training-free 기법이다(선배 연구). 논문에서 Omni-Steering(OURS)의 우수성을 입증하려면 기존 방법들과의 비교가 필요하다.

**내 담당은 비교군이다.** 기존 방법 4종을 동일 환경·동일 세팅으로 실행하여 정량 수치와 샘플별 출력을 확보한다. OURS 실행은 선배 담당이며 내 스코프가 아니다.

실행 매트릭스 (총 16 run):
```
모델 2종:    VideoLLaMA2-AV, Qwen2.5-Omni-7B
방법 4종:    Base(원본 디코딩), VCD-extended, AVCD, MAD   ← 전부 training-free decoding
벤치마크 2종: CMM, AVHBench
```
- CMM 지표: Visual Dominance / Audio Dominance / Language Dominance / Overall Acc
- AVHBench 지표: Video-Driven Audio Hallucination / Audio-Driven Video Hallucination / (AV Matching) / Overall Acc

내 출력은 나중에 선배의 OURS per-sample 결과와 **sample 단위로 join**되어, "OURS는 정답이고 기존 방법은 오답인 샘플"을 골라 논문 Figure(비디오 프레임 + 질문 + 방법별 답변 비교)로 만든다. 따라서 join 가능한 공통 키 유지가 필수다.

## 2. 최종 산출물

### D1. 정량 테이블
16 run의 벤치마크별·카테고리별 정확도 표. 논문의 메인 비교 테이블에 들어간다.

### D2. per-sample JSONL (join-ready)
- 공통 키: 벤치마크 원본의 video_id + question 식별자 (자체 넘버링 금지)
- 스키마 (전 방법 공통):
```json
{
  "sample_id": "...", "video_id": "...", "benchmark": "CMM|AVHBench", "category": "...",
  "question": "...", "ground_truth": "...",
  "method": "base|vcd_ext|avcd|mad", "model": "videollama2-av|qwen2.5-omni-7b",
  "prediction": "...", "correct": true,
  "internals": {
    "mad": {"w_v": 0.0, "w_a": 0.0, "w_av": 0.0},
    "avcd": {"dominant": "language|video|audio", "ead_skipped_ratio": 0.0}
  },
  "inference_time_s": 0.0, "seed": 42, "config_hash": "..."
}
```
- internals는 실패 원인 분석용: MAD가 틀렸을 때 weight 오배분 때문인지, AVCD가 틀렸을 때 dominance 오판/EAD skip 때문인지 판별하는 근거가 된다.

### D3. 사전 마이닝 리스트
- "MAD와 AVCD가 동시에 오답인 샘플" 목록 (+Base 오답 여부 태그). OURS 결과 없이도 만들 수 있고, 선배는 이 목록에서 OURS 정답 여부만 확인하면 Figure 후보가 완성된다.
- 후보 샘플에 대해 ffmpeg로 대표 프레임 4~8장 추출 + 방법별 답변 텍스트를 샘플당 폴더 하나로 패키징하는 스크립트 포함.

### D4. 부수 산출물
카테고리 세부표, 방법별 latency 표(전 방법 eager attention 통일 조건 명시), run manifest(재현 커맨드 기록).

## 3. 가용 리소스와 제약

| 항목 | 내용 |
|---|---|
| 로컬 | macOS(Apple Silicon) 1대. 7B 모델 forward 불가, CUDA 전용 패키지 설치 불가 |
| 서버 | A100 80GB × 1. **계정 발급 대기 중** — 발급 전까지 서버 작업 불가 |
| 서버에 이미 있는 것 (선배 제공 정보) | ① conda 환경 2종: `/home3/t202401082/.conda/envs/qwen-omni`, `/home3/t202401082/.conda/envs/videollama2` (복제해서 사용) ② OURS 코드+데이터셋: `/home3/t202401082/omni-steering` (직접 참조) |
| 작업 repo | github.com/hsmai/Hallucination |
| 참고 자료 | MAD 논문+공식 코드(github.com/top-yun/MAD), AVCD 논문+공식 코드(github.com/kaistmm/AVCD) |
| 마감 | 6일 |

여기서 나오는 설계 원칙:
1. **세팅 외부화**: 프롬프트, split, 디코딩 파라미터, 채점기, 경로 등 모든 세팅은 `configs/unified_settings.yaml`에서만 읽는다. 서버의 OURS 코드를 봐야 확정되는 값은 `UNKNOWN_pending_server`로 표기하고 임시 기본값은 MAD repo 세팅을 따른다. 서버 접속 후 yaml만 갱신하면 되는 구조.
2. **git 이동성**: 모든 산출물(코드/문서/스크립트/설정)을 repo에 커밋. 서버에서 `git clone` 한 번으로 전체 이전. 대용량은 .gitignore.
3. **로컬은 개발 전용**: 로컬에서는 모델 가중치·미디어 데이터를 다운로드하지 않는다(서버에 이미 있음). 벤치마크 QA json(경량)만 로컬로 받아 개발에 사용한다.

## 4. 통일 세팅 (unified_settings.yaml 항목)

| 항목 | 값 | 상태 |
|---|---|---|
| 디코딩 | greedy, temperature=0, 단일 run | 임시(MAD 논문 세팅) — 서버에서 OURS 코드 확인 후 최종 확정 |
| Attention 구현 | eager (FlashAttention/SDPA off), **전 방법 공통** | 확정 — AVCD가 full attention weight를 요구하며, 방법 간 공정성 |
| dtype / seed | bf16 / 42. 4bit 양자화 금지 | 확정 |
| MAD | γ=2.5 (전 데이터셋) | 확정 (논문) |
| AVCD | α^v=α^a=2.5(AVHBench)·0.5(기타), 마스킹 P=50%, 마지막 layer 제외 전 layer, EAD τ=0.6, plausibility β=0.1 | 확정 (논문). **CMM용 α는 논문에 없음** → 서버 게이트 단계에서 100샘플 그리드(0.5~3.0, 0.5 간격) 탐색 후 확정, 탐색 로그 보존 |
| VCD-extended | MAD 논문 Eq.10 형식 | 확정 |
| 프롬프트 템플릿 (CMM/AVHBench 각각) | | UNKNOWN_pending_server |
| AVHBench split (전체 vs validation) / CMM 카테고리·샘플 구성 | | UNKNOWN_pending_server |
| 채점기 | 전 방법을 **단일 채점기**로 재채점 | UNKNOWN_pending_server — omni-steering의 채점 스크립트를 그대로 이식하는 것이 최선 (join 시 correct 판정 기준이 OURS와 자동 일치) |
| 데이터 경로 프리픽스 | `/home3/t202401082/omni-steering/` 하위 | 서버에서 실경로 확인 후 기록 |

**세팅 통일의 최종 검증 기준**: 우리 하네스로 돌린 Base 수치가 선배가 보고한 Ours(Base) 수치와 일치해야 한다 — CMM Overall 73.6(VideoLLaMA2-AV)/73.1(Qwen), AVHBench Overall 77.4/76.8.

## 5. 방법 구현 참조

- **VCD-extended** (MAD 논문 Eq.10): 4-forward(clean, video왜곡, audio왜곡, 둘다왜곡) 후
  `(1+3α)·logit_vaq − α·logit_ṽaq − α·logit_vãq − α·logit_ṽãq`
- **MAD**: 생성 시작 전 1회, 입력 뒤에 modality query prompt("To answer this question, which modality is needed (audio, video, or both)?")를 붙여 forward → 다음 토큰 예측에서 'video'/'audio'/'both' 토큰의 logit만 추출해 softmax = (w_v, w_a, w_av) → 4개 contrastive branch를 γ·w로 가중합하여 매 스텝 디코딩. weight는 질문당 1회 추출 후 생성 내내 고정.
- **AVCD**: 마지막 query 토큰의 attention 분포를 layer 평균하여 dominant modality 판정 → 덜 dominant한 modality들의 attention 상위 P%(50%) 토큰을 zeroing(마지막 layer 제외 전 layer) → trimodal CD 결합식으로 4개 logit 조합 → entropy-guided adaptive decoding(원본 logit 엔트로피가 τ 미만이면 해당 스텝의 추가 forward를 skip) → adaptive plausibility constraint(β=0.1)로 후보 토큰 절삭. **full attention weight가 필요하므로 FlashAttention 사용 불가.**

## 6. 작업 단계

### 로컬 Phase (서버 계정 발급 전 — 지금 진행, 서버 무의존)

- **L1. 셋업 + 코드 정독**: repo 구조 생성, `third_party/`에 MAD·AVCD·VideoLLaMA2(audio_visual branch) clone → `docs/code_analysis.md` 작성 (VCD-ext가 MAD repo에 이미 구현돼 있는지 / AVCD의 modality 토큰 span 계산 방식 / MAD·AVCD 간 프롬프트·split·채점 불일치 / VideoLLaMA2의 eager attention 강제법) → `configs/unified_settings.yaml` 초안
- **L2. QA 데이터 + 로더**: 공개 소스에서 AVHBench/CMM의 QA json만 `data/qa/`로 확보 → 공통 데이터 로더(경로는 yaml 프리픽스 조립, 로컬은 파일 존재검사 skip) → `docs/data_report.md` (샘플 수·카테고리 분포)
- **L3. 하네스 (dry-run 완결)**: 방법 플러그인 구조의 공통 러너 + D2 스키마 JSONL 로깅 + 중단-재개(처리된 sample_id skip) + MockModel `--dry-run`으로 16조합 전부 end-to-end 검증 → 채점기(임시: MAD repo 로직 이식)·집계기(정량 테이블 자동 생성)까지 통과
- **L4. 방법 구현 + CPU 단위 테스트**: mad/vcd_ext/avcd 플러그인 구현. AVCD는 VideoLLaMA2용 공식 코드 이식 + **Qwen2.5-Omni용 신규 포팅**(공식 코드 없음 — 최대 리스크). 소형 더미 텐서로 수식·마스킹·EAD·plausibility를 pytest로 CPU 검증
- **L5. 서버 runbook 작성** (`docs/server_runbook.md`): 서버 계정 발급 당일 복붙 순서대로 실행할 문서 —
  1. 작업 repo clone
  2. conda env 복제: `conda create -n qwen-omni --clone /home3/t202401082/.conda/envs/qwen-omni` (videollama2 동일). 실패 시 fallback: `conda list -p <경로> --export`로 스펙 추출 후 신규 생성. 복제 후 `conda list` 스냅샷을 docs/에 커밋
  3. **OURS 세팅 추출** (`/home3/t202401082/omni-steering` 정독): 인퍼런스 스크립트의 프롬프트·디코딩 파라미터, 데이터 로더의 split·샘플 수, 채점 스크립트(→ 이식), 실행 커맨드, 데이터 실경로, Ours 결과 JSON 존재 여부 → yaml의 UNKNOWN 항목 전부 확정
  4. 모델 가중치 위치 확인(HF 캐시 등), 없을 때만 다운로드
  5. run_smoke.sh → run_gate.sh → run_full.sh 순 실행

### 서버 Phase (계정 발급 후)

- **S1. 개통일**: runbook 1~4 수행 → 스모크(조합당 5샘플) → 방법별 처리 속도 실측 → 전체 소요시간표 작성
- **S2. 정합성 게이트** (방법별 100~200샘플):
  1. **Base ≈ Ours(Base) 수치 일치** — 세팅 통일의 핵심 검증. 어긋나면 yaml 항목별(프롬프트/split/attention/dtype) diff로 원인 추적 후에만 진행
  2. 논문 수치 재현 확인(±2%p): MAD@AVHBench(VideoLLaMA2-AV)≈79.4, AVCD@AVHBench(VideoLLaMA2)≈72.15(전체셋 기준)/81.95(validation 기준) — 이 대조가 AVHBench split 판정을 겸함
  3. AVCD의 CMM용 α 그리드 탐색
- **S3. 본 실행**: 16 run을 모델 단위로 묶어 순차 실행(모델 로드 1회당 4방법), nohup 야간 무인, 완료분부터 채점 병행. GPU 시간 부족 시 우선순위: ① AVHBench 전 방법(정성 샘플의 주 무대) ② CMM의 MAD/AVCD ③ CMM의 VCD-ext
- **S4. 정리·납품**: D1 테이블, D2 JSONL 검수, D3 마이닝 리스트+프레임 패키징, D4 manifest → 선배에게 전달

## 7. 6일 일정

| Day | 내용 | 의존성 |
|---|---|---|
| 1 | L1 + L2 | 없음 |
| 2 | L3 | 없음 |
| 3 | L4 + L5 | 없음 |
| 4 | S1 + S2 | 서버 계정 |
| 5 | S3 | 서버 |
| 6 | S4 | 서버 (D3는 OURS 결과 없이도 완성 가능) |

계정이 늦어지면 S3를 우선순위대로 잘라내고, 일찍 나오면 L4~L5와 S1을 병행한다(세팅 추출을 앞당겨 확정 세팅 위에서 개발).

## 8. 선배와의 인터페이스

| 시점 | 항목 |
|---|---|
| 지금 | 서버 계정 발급 요청 + **두 경로(.conda/envs/*, omni-steering)에 내 계정 읽기 권한이 있는지 확인** |
| 즉시 합의 | join 규약: sample_id 체계(원본 video_id+question), D2 스키마 필드명, 공통 채점기(=omni-steering의 채점 스크립트) |
| 서버 접속 후 | omni-steering에 없는 정보(예: 실행 커맨드 미기록)만 선별 질문 |
| Day 5~6 | OURS per-sample 결과 JSON (omni-steering 내 결과 파일로 대체 가능하면 요청 불필요) |

## 9. 리스크

1. **서버 계정/권한 지연** — 로컬 L1~L5는 무의존이므로 계속 진행. 권한 문제는 선배에게 chmod/그룹 설정 요청
2. **Qwen2.5-Omni용 AVCD 포팅** — 공식 코드 없음. L4에서 modality span·attention 접근 구조부터 검증. 실패 시 fallback: 해당 칸을 "공식 코드 미지원"으로 표기하고 VideoLLaMA2 결과만 사용
3. omni-steering 내부가 예상과 다를 가능성(채점 스크립트 부재 등) — S1에서 확인, 부재 항목만 선배 질문
4. AVHBench split 불일치 — S2 게이트에서 판정 후 yaml에 기록
5. conda env 복제 실패(절대경로 하드코딩 등) — export fallback으로 우회
6. GPU 시간 부족 — S3 우선순위 컷 + 야간 무인 실행으로 흡수
