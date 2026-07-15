# 논문·실험자료 발췌 (재열람 불필요용 단일 소스)

> 2026-07-15 사용자 제공 4개 PDF에서 1회 발췌. **이후 PDF를 다시 읽지 말고 이 문서를 근거로 사용한다.**
> 원본: `~/Downloads/Experiments.pdf`, `~/Downloads/Omni-steering.pdf`(50p 슬라이드),
> `~/Downloads/Hallucination/MAD.pdf`(CVPR2026, arXiv:2601.21181), `~/Downloads/Hallucination/AVCD.pdf`(NeurIPS2025).

---

## 1. MAD 논문 (CVPR 2026)

### 수식 (코드와 대조 검증 완료)

- **Eq.9 (MAD)** — 4-branch. 전개하면 branch별 계수가 repo 코드(`mm_contrast_decode`)와 **정확히 일치**:
  - `logit_vaq`(VA): `2 + 2γ·w_av`
  - `logit_vãq`(V만, audio 제거): `1 + γ(w_v − w_av)`  ↔ 코드 `1 − γ(p_both − p_video)` ✓
  - `logit_ṽaq`(A만): `1 + γ(w_a − w_av)` ✓
  - `logit_ṽãq`(T만): `−γ(w_v + w_a)` ✓
  - **perturbation(˜)의 실체 = modality 제거(absence)** — 논문 Eq.7 표기와 코드 모두 ablation 프롬프트
- **Eq.6 (weight)**: `[w_av, w_v, w_a] = softmax([z_both, z_video, z_audio])`, head 프롬프트 1회 forward, 질문당 1회 고정 ✓
- **Eq.10 (VCD-Extended 베이스라인 정의)**: `(1+3α)·logit_vaq − α·logit_ṽaq − α·logit_vãq − α·logit_ṽãq`
  = repo `mm_default_cd` 계수와 일치. **VCD-ext = MAD식 modality-ablation branch로 확정** (blueprint 충돌 해소)
  - ⚠ VCD-ext의 α 값은 논문·코드 어디에도 명시 없음 (`mm_default_cd` default 0.5) → 게이트에서 {0.5, 2.5} 대조

### 세팅

- temperature=0 (deterministic), **γ=2.5 전 데이터셋** (100샘플, 0.5~3.0 step 0.5 그리드로 결정)
- modality query prompt: “To answer this question, which modality is needed (audio, video, or both)” — 문구 변형에 강건(std 0.26~0.31%)
- 벤치마크: CMM(Visual/Audio/Language Dom + Overall), AVHBench(VdAH/AdVH + Overall — **AV Matching·Captioning 없음**)
- 모델: VideoLLaMA2-AV-7B, Qwen2.5-Omni-7B

### Table 1 (= Experiments.pdf 상단 표와 동일 — 우리 재현 목표치)

| Model | CMM V/A/L/Overall | AVH VdAH/AdVH/Overall |
|---|---|---|
| VideoLLaMA2-AV (Base) | 71.8 / 80.0 / 68.8 / 73.5 | 75.7 / 79.0 / 77.4 |
| + VCD-Extended | 71.3 / 83.3 / 74.8 / 76.4 | 66.0 / 74.8 / 70.4 |
| + AVCD | 71.8 / 84.0 / 71.5 / 75.8 | 78.3 / 80.3 / 79.3 |
| + MAD | 82.3 / 84.3 / 77.5 / 81.3 | 79.7 / 79.1 / 79.4 |
| Qwen2.5-Omni-7B (Base) | 64.5 / 72.3 / 81.3 / 72.7 | 73.0 / 80.7 / 76.9 |
| + VCD-Extended | 62.5 / 71.3 / 84.5 / 72.8 | 70.3 / 77.1 / 73.7 |
| + AVCD | 66.3 / 72.8 / 81.0 / 73.3 | 75.8 / 79.7 / 77.8 |
| + MAD | 76.8 / 84.3 / 83.3 / 81.4 | 78.7 / 84.4 / 81.6 |

- 부록: MAD의 modality weight 분석(AVHBench V→A에서 w_a 0.615 우세 등), latency 표(ms/token: VCD-ext 3564/4432, AVCD 4811/9490, MAD 3572/6701 — VideoLLaMA2/Qwen)

## 2. AVCD 논문 (NeurIPS 2025)

### 수식 (코드와 대조 검증 완료)

- **Eq.10 (trimodal CD)**, language dominant·αv=αa=α일 때:
  `(2+2α)·logit(v,a,l) + 1·logit(¬v,a,l) + 1·logit(v,¬a,l) − 2α·logit(¬v,¬a,l)`
  ↔ 코드 `(2+2α)·orig − 2α·out1(둘 다 마스킹) + out2 + out3` **일치** ✓
  (일반형: `(2+αv+αa)`, `(1−αv+αa)`, `(1+αv−αa)`, `−(αv+αa)`)
- **Eq.11** = MAD의 Eq.10과 동일한 (1+3α) 형식 — AVCD도 이를 "conventional CD 확장"으로 대조 실험 (AVH full 70.94)
- dominance: 마지막 query 토큰 attention을 modality span별 합산, layer 평균(마지막 layer 제외) → argmax
- attentive masking: 층 누적 A_QK 평균의 **top P=50%** 토큰 zeroing, **마지막 layer 제외 전 layer**
- EAD: 원본 logit 엔트로피 < **τ=0.6** 이면 CD skip
- **plausibility: β=0.1**, V_head 밖 토큰은 **−inf** (Eq.17–18) — ⚠ 공개 코드는 β=0.2, fill −1e-4 (코드≠논문, 양쪽 유지)

### 세팅·수치

- α: 100샘플 0.5~3.0 step 0.5 그리드 → **AVHBench 2.5, 그 외 데이터셋 0.5** (CMM은 논문에 없음 → 그리드 필요, blueprint와 일치)
- αv=αa (dominance가 video/audio 균형이라는 분석 기반); αv=2.0,αa=2.5로 바꿔도 val 81.95 동일
- masking ratio ablation (AVH val): 25%→80.98, **50%→81.95**, 75%→80.00, 100%→80.00
- split: "전체 데이터셋=test, 최초 공개분=validation" → **AVH full test 72.15 / val 81.95** (VideoLLaMA2, blueprint 게이트 수치와 일치)
- τ ablation: τ=0.8이면 VCD보다 빠르면서 base보다 정확 (78.05→80.98)
- component ablation (val): random-dominant CD 74.15 → +dominance masking 79.02 → +Eq.10 81.95 → +EAD 81.95(3.1s/token)
- FlashAttention 미사용 (full attention weight 필요) — eager 통일 근거
- AV-LLM에서 language dominance가 70% (200샘플 전부 language dominant)

## 3. Omni-Steering 슬라이드 (선배 연구, 26.07.13 버전)

- **방법**: 답변 토큰 hidden state에서 "질문받지 않은 modality가 주입한 성분" 제거.
  `h1 = full(a,v,t)`, `h2 = 대상 modality 마스킹`, `d_c = h1 − h2`, `h ← h − α·d_c`.
  마스킹 = **attention key masking** (모든 query가 해당 modality key를 못 보게 −inf).
  실험 세팅: **Layer 22, α=1.0**. (이전 버전: 4-forward AVT/A'VT/AV'T/A'V'T + orthogonal projection + asym gate)
- 남은 문제(슬라이드): adaptive layer 선택, polluted modality 자동 판정(현재 keyword 방식→모델 self-decision 전환 예정), language dom 해결
- 슬라이드 p45 실험표 = MAD 논문 Table 1 + Ours 행 (아래 §4)

## 4. Experiments.pdf — 진행 상태 해석

- **상단 표 = MAD 논문 Table 1 그대로** (비교군 숫자는 논문 발표치. 우리 재현 목표 ±2%p)
- **하단 표(볼드 행) = 선배가 실제 완료한 실험**:

| | CMM V/A/L/Overall | AVH VdAH/AdVH/Overall |
|---|---|---|
| **Ours(Base)** VideoLLaMA2-AV | 72.3 / 81 / 68.8 / **74** | 75.9 / 79 / **77.4** |
| **Ours** (VideoLLaMA2-AV) | 85 / 90 / 68 / 81 | 78.4 / 82.9 / 80.7 |
| **Ours(Base)** Qwen | 68.8 / 69.5 / 81 / **73.1** | 73.3 / 80.4 / **76.8** |
| **Ours** (Qwen) | 83.5 / 89.2 / 82 / 84.9 | 81 / 85.1 / 83.1 |

- ⚠ 슬라이드 p45의 Ours(Base) VideoLLaMA2는 72.0/80.2/68.5/**73.6** (Experiments.pdf는 74) — 두 버전 존재.
  blueprint의 "73.6"은 슬라이드 버전. **게이트 목표: CMM Overall 73.6~74.0 / 73.1, AVH 77.4 / 76.8**
- **정황(가설)**: Ours(Base) ≈ MAD 논문 Base(77.4=77.4, 76.8≈76.9, 73.6/74≈73.5, 73.1≈72.7)
  → 선배 파이프라인이 MAD 프로토콜(QA.json에서 AV task 제외 split + MAD 채점 방식)을 따를 가능성이 높음.
  **단, 이는 정황일 뿐 확정이 아니다.** 로컬 개발은 이 가설을 임시 기본값으로 진행하되,
  모든 항목은 서버의 OURS 코드 확인 전까지 UNKNOWN_pending_server 상태를 유지한다.

## 5. blueprint 충돌 정리 (전부 **잠정** — 서버 OURS 확인 후 최종 확정)

| 항목 | 이전 상태 | 정리 결과 (잠정) |
|---|---|---|
| VCD-ext 방식 (왜곡 vs 제거) | 이중성 | MAD 논문 Eq.10 근거로 **modality 제거를 잠정 채택** (α 미정: {0.5,2.5} 게이트 대조) |
| AVCD β | 코드 0.2 vs blueprint 0.1 | 논문은 0.1(−inf), 코드는 0.2(−1e-4) — 양쪽 유지, 게이트에서 val 81.95 재현되는 쪽 채택 |
| AVHBench split | 3종 후보 | MAD 프로토콜(AV task 제외 3,426)이 정황상 유력 — temp_default로만 사용. AVCD 논문 수치 대조시에만 full(72.15)/val(81.95) 사용 |
| CMM 구성 | all vs over-reliance | 지표가 V/A/L Dominance → over-reliance_unimodal_priors 카테고리가 유력. L2에서 데이터로 검증, 서버에서 확정 |
| CMM용 AVCD α | 미정 | 논문 규칙 "AVHBench 외 0.5" → temp 0.5 + 그리드 유지 |
| 게이트 목표치 | Base만 | MAD 논문 Table 1 전체(§1) + Ours(Base)(§4)로 확장 |

## 6. 핵심 목표·산출물 정의 (2026-07-15 사용자 지시로 명문화)

**목표**: Ours(Omni-Steering) 대비 우수성 입증 자료 확보. 구체적으로:

1. **정량**: 비교군 4종(Base/VCD-ext/AVCD/MAD) × 2모델 × 2벤치마크(CMM, AVHBench)를
   **서버에 세팅된 환경·동일 데이터·동일 벤치마크에서 직접 실행**하여 결과 확보
   = Experiments.pdf의 **비볼드 수치들(= MAD 논문 Table 1)을 우리 손으로 재생산** (MAD 기준 main Table 1 확보)
2. **정성**: MAD 논문 15~16p 스타일의 **비교 가능 샘플**(같은 입력에 대한 방법별 답변 대조)
   = blueprint D3(마이닝 리스트 + 프레임 패키징)와 동일 항목. per-sample JSONL(D2)이 그 재료

**AVCD 재구현 갭 2건 (사용자와 교차 확인됨, blueprint L4-3·리스크 §9-2와 일치)**:

- MAD 논문 Table 1의 AVCD 수치(CMM 포함, Qwen 포함)는 **MAD 저자들의 비공개 재구현**으로 산출된 것.
  MAD repo에는 base/MAD 스크립트만 있고 AVCD 재구현 코드는 없다. AVCD 공식 repo는
  VideoLLaMA2 전용 + 벤치마크도 다름(MUSIC-AVQA/AVHBench, CMM 없음).
- 따라서 우리가 메워야 할 갭:
  1. **AVCD 디코딩 로직을 MAD식 하네스(CMM 평가 파이프라인)에 이식**
  2. **AVCD의 Qwen2.5-Omni 버전 신규 포팅** (attention 접근 + modality 토큰 span 동적 계산 포함)
- 함의: 우리 AVCD 구현이 Table 1의 AVCD 수치와 정확히 일치할 보장은 없음(저자 재구현의 세부 미공개).
  Table 1 AVCD 수치는 ±2%p **참고 기준**이며, 불일치 시 원인 분석 기록 후 우리 구현 수치를 보고.
  실패 시 fallback: 해당 칸 "공식 코드 미지원" 표기 (blueprint §9-2).

**서버 의존성**: 실제 16 run 실행은 서버 계정 발급 후에만 가능. 그 전 로컬 작업(L2~L5)은
데이터 로더·하네스·방법 구현·CPU 테스트·runbook 등 서버 무의존 준비에 한정한다.
