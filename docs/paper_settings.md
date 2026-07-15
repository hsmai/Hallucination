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
- **핵심 정황**: Ours(Base) ≈ MAD 논문 Base(77.4=77.4, 76.8≈76.9, 73.6/74≈73.5, 73.1≈72.7)
  → **선배 파이프라인은 MAD 프로토콜(QA.json에서 AV task 제외 split + MAD 채점 방식)을 따를 가능성이 매우 높음.**
  로컬 개발은 MAD 프로토콜 기준으로 확정 진행, 서버에서 OURS 코드로 최종 검증만 수행.

## 5. blueprint 충돌 해소 요약

| 항목 | 이전 상태 | 해소 결과 |
|---|---|---|
| VCD-ext 방식 (왜곡 vs 제거) | 이중성 | **MAD Eq.10 = modality 제거로 확정** (α만 미정: {0.5,2.5} 게이트 대조) |
| AVCD β | 코드 0.2 vs blueprint 0.1 | 논문도 0.1(−inf). 코드(0.2, −1e-4)와 상이 — faithful_mode(코드) 기본, 논문 모드 옵션. 게이트에서 val 81.95 재현되는 쪽 채택 |
| AVHBench split | 3종 후보 | MAD 프로토콜(AV task 제외 3,426) 사실상 확정. AVCD 논문 수치 대조시에만 full(72.15)/val(81.95) 사용 |
| CMM 구성 | all vs over-reliance | 지표가 V/A/L Dominance → **over-reliance_unimodal_priors 카테고리**(의 sub_category)로 사실상 확정. L2에서 데이터로 검증 |
| CMM용 AVCD α | 미정 | 논문 규칙 "AVHBench 외 0.5" → temp 0.5 + 그리드 유지 |
| 게이트 목표치 | Base만 | MAD 논문 Table 1 전체(§1) + Ours(Base)(§4)로 확장 |
