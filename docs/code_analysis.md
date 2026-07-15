# 코드 분석: MAD / AVCD / VideoLLaMA2(audio_visual)

> Phase L1 산출물. 근거는 third_party/에 clone한 실제 코드이며, 커밋은 다음에 고정되어 있다
> (`scripts/setup_third_party.sh`로 재현):
>
> | repo | branch | commit |
> |---|---|---|
> | top-yun/MAD | main | `e6f5072` (2026-06-22) |
> | kaistmm/AVCD | main | `92aeaf1` (2025-11-03) |
> | DAMO-NLP-SG/VideoLLaMA2 | audio_visual | `8cedfbd` (2024-11-04) |
>
> 라인 번호는 위 커밋 기준.

---

## 1. MAD (third_party/MAD)

### 1.1 디렉토리 구조: `VideoLLaMA2/` vs `qwen-omni/`

| | `MAD/VideoLLaMA2/` | `MAD/qwen-omni/` |
|---|---|---|
| 모델 코드 | videollama2 패키지 전체 fork를 동봉(모델/프로세서/컨버세이션 포함). 디코딩 함수가 패키지 `videollama2/__init__.py`에 통합 | 모델 fork 없음. HF `transformers`의 `Qwen2_5OmniForConditionalGeneration` + `Qwen2_5OmniProcessor`를 직접 사용, 디코딩은 `utils.py` 한 파일 |
| 디코딩 진입점 | `videollama2/__init__.py`: `mm_infer`(base), `mm_contrast_decode`(MAD), `mm_default_cd`(VCD-ext형), `mm_logit`, `mm_test` | `utils.py`: `mm_contrast_decode_qwen`(MAD). base는 `eval_batch*.py`에서 `model.generate` 직접 호출 |
| forward 방식 | `model(...)` (videollama2 wrapper가 멀티모달 임베딩 준비) | `model.thinker(...)` (talker 미사용, `disable_talker()` 옵션) |
| 실행 스크립트 | `eval_batch{,_mad,_cmm,_cmm_mad}.py` (accelerate) | 동일 구성 + `batch_*.sh` (γ=2.5 기록됨) |
| 채점 | `score.py`, `score_cmm.py` | **VideoLLaMA2 쪽과 바이트 단위 동일 파일** |

### 1.2 MAD 디코딩 구현 위치

- VideoLLaMA2: `MAD/VideoLLaMA2/videollama2/__init__.py` — `mm_contrast_decode` (L257–456)
- Qwen2.5-Omni: `MAD/qwen-omni/utils.py` — `mm_contrast_decode_qwen` (L8–163)
- 호출부: `eval_batch_mad.py`(AVHBench), `eval_batch_cmm_mad.py`(CMM) 각 디렉토리

### 1.3 Modality query prompt와 weight 추출

- 프롬프트 문자열 (`VideoLLaMA2/config.py` L1, `qwen-omni/utils.py` L5 동일):
  ```
  MODALITY_QUERY_PROMPT = "To answer this question, which modality is needed (audio, video, or both): "
  ```
- head 프롬프트 구성: `<video>\n` + `"Question: " + question + "\n"` + MODALITY_QUERY_PROMPT
  (question에는 answer-format suffix가 이미 붙은 상태로 들어감)
- 추출 (`__init__.py` L378–388, `utils.py` L89–99): head 프롬프트 1회 forward → 마지막 위치 logits에서
  `tokenizer.encode("audio")[0]`, `"video"`, `"both"`의 **첫 토큰 id** logit 3개만 뽑아 softmax
  → `(audio_prob, video_prob, both_prob)`. **생성 시작 전 1회 계산 후 생성 내내 고정** (지시서와 일치).

### 1.4 4-branch 가중합 수식 (⚠ 지시서 요약보다 구체적 — 코드 기준으로 구현할 것)

4개 branch는 **입력 왜곡이 아니라 modality 제거(ablation)** 프롬프트다:
`[VA(비디오+오디오), V(비디오만), A(오디오만), T(텍스트만)]`.
각 branch를 개별 KV cache로 유지하며 매 스텝 동일 토큰을 공급한다.

가중치 (`__init__.py` L418–423, `utils.py` L125–135 — 두 모델 동일 수식):

```
w_VA = 2 + 2γ·p_both
w_V  = 1 − γ·(p_both − p_video)
w_A  = 1 − γ·(p_both − p_audio)
w_T  = −γ·(p_video + p_audio)
next_token = argmax( w_VA·logit_VA + w_V·logit_V + w_A·logit_A + w_T·logit_T )
```

- 가중치 총합은 γ, p에 무관하게 항상 4.
- γ: 논문/셸 스크립트 기준 **2.5** (`batch_avhbench_mad.sh`, `batch_cmm_mad.sh`, qwen CMM MAD의 argparse default 2.5).
  ⚠ python argparse default는 0.5인 파일이 많음 — 셸 스크립트가 2.5를 명시적으로 넘긴다. 재현 시 2.5 사용.
- greedy(argmax), `do_sample=False`.

### 1.5 데이터 로더와 프롬프트 템플릿

**AVHBench** (`eval_batch*.py`):
- `QA.json` (전체 6,408) 로드 후 `if 'AV' in task: continue` → **AV Matching / AV Captioning 제외**,
  Video-driven Audio Hallucination 2,290 + Audio-driven Video Hallucination 1,136 = **3,426샘플 사용**
- 미디어: `videos/{video_id}.mp4`, audio는 경로 치환 `videos→audios`, `.mp4→.wav`
- `max_new_tokens=1` (yes/no 첫 토큰만 생성)

**CMM** (`eval_batch_cmm*.py`):
- 단일 json `all_data_final_reorg.json`. 필드: `video_path, question, answer, category, sub_category, modality, granularity, correlation_type`
- `--category` 필터: `all | inter-modality_spurious_correlation | over-reliance_unimodal_priors` (**default: over-reliance_unimodal_priors** ⚠ 전체가 아님)
- audio 경로: video와 동일 파일명 `.wav`
- 출력은 JSONL(`pred`/`answer` 키)

**프롬프트 suffix — MAD repo 내부에서도 불일치 (그대로 기록):**

| 스크립트 | suffix |
|---|---|
| VideoLLaMA2 base AVHBench (`eval_batch.py`) | `" Answer only 'Yes' or 'No'."` |
| VideoLLaMA2 MAD AVHBench (`eval_batch_mad.py`) | `" Answer yes or no."` |
| VideoLLaMA2 base CMM (`eval_batch_cmm.py`) | `" Only yes/no. No extra text."` |
| VideoLLaMA2 MAD CMM / qwen 전 스크립트 | `" Answer only 'Yes' or 'No'. Do not include any explanation."` (=`ANSWER_QUERY_PROMPT`) |

**Qwen system prompt** (base/MAD 공통): `"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."`
VideoLLaMA2.1-AV(qwen2 backbone)는 chat template 기본값 외 별도 system message 없음(`<<SYS>>` 블록은 mistral/mixtral 계열에만 삽입).

**Qwen branch 구성**: `av` = video 파일 + `use_audio_in_video=True`, `v` = 동일 video + `use_audio_in_video=False`, `a` = audio 파일, `t` = 텍스트만.

### 1.6 채점 규칙

`score.py` (AVHBench용, 두 모델 디렉토리 동일 파일):
- `extract_answer`: 소문자화 후 정규식 — **yes 패턴(`yes|true|correct|affirmative`)을 no 패턴보다 먼저 검사** → 둘 다 있으면 yes. 패턴 없으면 첫 단어를 normalize
- `normalize_answer`: lowercase/strip, 끝 문장부호 제거, `y/true/1/correct→yes`, `n/false/0/incorrect→no`, 그 외 substring `yes`/`no` 검사
- (video_id, question) 중복 제거, `ERROR:` 예측 제외
- task별 accuracy 출력, **Overall은 AV Captioning·AV Matching 제외** 합산

`score_cmm.py`: JSONL 입력, `score.py`의 extract/normalize 재사용, overall + `sub_category`별 accuracy.

### 1.7 VCD-extended 구현 여부 — **있음(부분)**

- `MAD/VideoLLaMA2/videollama2/__init__.py` — `mm_default_cd` (L978–1177):
  branch `[VA, V, A, T]`에 고정 가중치 **`[1+3γ, −γ, −γ, −γ]`** → 지시서의
  `(1+3α)·logit_vaq − α·… − α·… − α·…` 계수와 정확히 일치.
- ⚠ 단, 지시서는 "왜곡(distorted)" 입력 4-forward로 기술했으나 **MAD 코드는 modality 제거 branch**를 쓴다
  (ṽaq→A branch, vãq→V branch, ṽãq→T branch에 대응하는 구조).
- ⚠ **어떤 eval 스크립트도 `mm_default_cd`를 호출하지 않는다** (dead code). γ 기본값 0.5.
- ⚠ **qwen-omni용 VCD-ext는 없다** → 신규 작성 필요 (`mm_contrast_decode_qwen`의 branch/캐시 구조 재사용 + 가중치만 교체하면 됨).
- 참고: AVCD repo의 `videollama2/__init__VCD.py`에는 **diffusion noise(step 500) 왜곡 기반** 진짜 VCD 구현이 있으나,
  패치된 `transformers/generation/utils.py`(`use_cd=True` kwarg)에 의존하며 그 패치는 repo에 없음 → 그대로 실행 불가.
- **결정 필요(서버에서 OURS 세팅 확인 시)**: VCD-ext를 (a) MAD식 modality-제거 branch로 할지 (b) AVCD식 노이즈 왜곡으로 할지.
  임시 기본값은 MAD repo 세팅(=modality 제거, `mm_default_cd` 계수)을 따른다.

---

## 2. AVCD (third_party/AVCD)

공식 구현은 VideoLLaMA2.1-7B-AV 전용. 핵심 로직은 fork된 백본 `videollama2/model/qwen.py`(Qwen2 modeling)와 `videollama2/__init__.py`의 `mm_infer(generate_long=True, use_AVCD=...)` 경로에 있다.

### 2.1 실행 경로 주의사항

- `mm_infer`의 `generate_long=False` 경로는 `model.generate(..., use_AVCD=...)`를 호출하는데, 이는 **레포에 포함되지 않은 패치된 transformers generation/utils.py를 요구**한다(주석 L110). 
- 실제 AVHBench 스크립트(`inference_AVH_val.py`, `inference_AVH_test.py`)는 `generate_long=True, cd_alpha=2.5`를 사용 — **KV cache 없이 매 스텝 전체 시퀀스를 re-forward**하는 수동 루프(느림). EAD 스킵 시 스텝당 1 forward, 아니면 4 forward.
- 우리 구현은 이 수식을 유지하되 KV cache 구조로 재작성한다(단, masking이 프롬프트 토큰 attention에 걸리므로 CD branch는 매 스텝 full forward가 필요한 구조인지 검증 필요 — threshold/dominance는 orig forward에서만 계산되므로 orig branch는 캐시 가능).

### 2.2 Dominance 계산 위치와 방식

`videollama2/model/qwen.py` `Qwen2Model.forward` L1176–1223 (modality=None인 orig forward에서만):

1. 각 layer의 attention에서 **마지막 query 토큰의 attention 행**(head 평균)을 수집 (`Qwen2SdpaAttention.forward` L782, L799가 `[last_query]`로 반환)
2. modality span별 합을 span 길이로 정규화: `video_sum/676`, `audio_sum/1496`, `lang_sum/(last−2186)`
3. **마지막 layer(idx 27) 제외**하고 layer 평균 → `avg_dominance = sorted([...], reverse=True)`, `[0][0]`이 dominant
   (⚠ 합산은 27개 layer인데 나눗셈은 `len(layer_attention)=28` — 코드 그대로의 사소한 버그, 순위에는 영향 없음)

### 2.3 Modality 토큰 span 계산 방식 — **하드코딩** (Qwen 포팅의 핵심 재료)

`qwen.py` L710–711:
```python
start = 14                 # system/head prompt 토큰 수
last  = 14 + 676 + 1496    # = 2186
```
- video span = `[14, 690)` (676 = VideoLLaMA2.1-AV의 STC connector 출력 토큰 수)
- audio span = `[690, 2186)` (1496 = BEATs 오디오 토큰 수)
- language(question) span = `[2186, seq_len)`
- 이 값은 **VideoLLaMA2.1-7B-AV + va=True + 해당 chat template에서만 유효**. 근거: `videollama2_arch.py::prepare_inputs_labels_for_multimodal`이 프롬프트의 단일 `<video>` placeholder 위치에 `concat([video_feat(676), audio_feat(1496)])`를 삽입하는 구조 (L206–335).
- **Qwen2.5-Omni 포팅 시**: Qwen은 placeholder pad 토큰(`<|AUDIO|>`/`<|VIDEO|>` 계열)이 `input_ids`에 확장되어 남으므로, token id로 span을 **동적으로** 찾을 수 있다. `use_audio_in_video=True`면 audio/video 토큰이 시간순으로 **interleave**되므로 연속 구간이 아닌 **boolean mask** 방식으로 구현해야 한다.

### 2.4 Attentive masking 구현

`qwen.py` `Qwen2SdpaAttention.forward` L713–779:
- 이 클래스는 이름과 달리 SDPA를 쓰지 않고 **수동으로 full attention matrix를 fp32로 계산**(eager 동작). prefill(q_len>1)에서만.
- threshold: orig forward에서 layer 0–26의 last-query attention 누적 평균의 **0.5-quantile(중앙값)** → 상위 50% 컷 (`qwen.py` L1212–1213). P=50%는 논문과 일치.
- CD branch forward에서 `modality`/`threshold`가 전달되면, **layer_idx < 27에서만**:
  `av_mask = (last_query_attn <= threshold)` → 지정 modality span에서 **attention 상위 50% 토큰의 열을 0으로** 만들고 행별 재정규화(`clamp(min=1e-6)`).
- modality 문자열: `"V"`(video span만), `"A"`(audio), `"L"`(language), `"VA"`, `"LA"`, `"LV"` 조합.

### 2.5 결합식·EAD·plausibility (`videollama2/__init__.py` L154–235)

dominant에 따른 branch 선택:

| dominant | out1 (마스킹) | out2 | out3 |
|---|---|---|---|
| language | VA | A | V |
| video | LA | A | L |
| audio | LV | V | L |

```
entropy(softmax(orig_logits)) < 0.6  →  next = orig_logits  (EAD skip, τ=0.6)
그 외:
  contrastive = (2+2α)·orig − 2α·out1 + out2 + out3          # α = cd_alpha
  cutoff = log(β) + max(orig_logits)                          # β = 0.2 (코드값)
  next = contrastive.masked_fill(orig_logits < cutoff, -1e-4)
```

⚠ **코드 vs 논문/지시서 충돌 (보고 사항)**:
1. plausibility **β: 코드 0.2, blueprint·지시서 0.1** → 작업 원칙에 따라 코드값 0.2를 임시 기본값으로 두고 yaml로 노출, 서버 게이트에서 결정
2. masked_fill 값이 `-inf`가 아니라 **`-1e-4`** — implausible 토큰이 완전히 배제되지 않는 코드 그대로의 동작. 동일 재현을 위해 코드대로 구현하되 플래그로 `-inf` 전환 가능하게 함
3. α(cd_alpha): 함수 default 0.5, **AVHBench 스크립트는 2.5 명시** (blueprint의 "AVHBench 2.5, 기타 0.5"와 정합)

### 2.6 AVHBench split 정의 (`AVCD/json/`)

| 파일 | 샘플 수 | 구성 |
|---|---|---|
| `AVH_val.json` | 205 | VdAH 80 / AdVH 66 / AV Matching 59 |
| `AVH_test.json` | 5,302 | VdAH 2,290 / AdVH 1,136 / AV Matching 1,876 |
| `QA.json` | 6,408 | 위 + AV Captioning 1,106 (MAD가 쓰는 파일과 동일 구성) |

- 필드: `video_id, task, text, label` (MAD와 동일 스키마 → join 키 호환)
- AVCD 논문 수치 재현 대조(blueprint S2): 72.15(전체)/81.95(val) → split 판정은 서버 게이트에서
- AVCD 프롬프트: `text + " Answer yes or no."`
- AVCD 채점 (`videollama2/eval/eval_acc.py`): pred에 "yes" 또는 "no"가 있고 `answer.lower() in pred.lower()`면 정답 — MAD 채점기와 **다름**

---

## 3. VideoLLaMA2-AV (third_party/VideoLLaMA2, audio_visual branch)

### 3.1 모델 로드

- `videollama2.model_init(model_path, **kwargs)` → `model/__init__.py::load_pretrained_model` → 임의 kwargs가 `from_pretrained`로 전달됨
- 모델: `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` (Qwen2 backbone, `model_type=videollama2_qwen2`)
- processor dict: `{'image', 'video'(process_video, num_frames=8 기본), 'audio'(process_audio_file)}`; `processor['video'](path, va=True)` → `{'video': tensor, 'audio': tensor}` dict

### 3.2 Eager attention 강제 방법

- `load_pretrained_model(..., use_flash_attn=True)`이면 `attn_implementation='flash_attention_2'` 주입 (L79–80). 
- **eager 강제: `model_init(model_path, attn_implementation="eager", torch_dtype=torch.bfloat16)`** — kwargs가 그대로 `from_pretrained`에 전달되므로 동작. `output_attentions=True`도 eager에서만 실제 attention을 반환.
- MAD 스크립트들은 base=flash_attn, MAD=sdpa, CMM MAD=flash_attn으로 **제각각** → 우리 하네스에서 전 방법 eager로 통일 (통일 조건은 yaml에서 강제)

### 3.3 audio/video 토큰 시퀀스 배치 구조

`videollama2_arch.py::prepare_inputs_labels_for_multimodal` (L206–):
- 프롬프트 텍스트에는 placeholder 토큰 1개(`<video>`=−201 계열 MODAL_INDEX_MAP 음수 id)만 존재
- va=True(dict 입력)면 `mm_features = concat([video_features, audio_features], dim=0)` (L262) — **video 먼저, audio 뒤**
- placeholder 위치에서 input_ids를 잘라 `[앞 텍스트 임베딩; video 676; audio 1496; 뒤 텍스트 임베딩]`으로 조립
- 따라서 시퀀스 = `[head prompt(14 tokens); video(676); audio(1496); question+generation prompt]` — AVCD 하드코딩 span의 근거
- MAD의 `mm_logit`(L197–199)은 같은 span을 동적으로 계산하는 예시 코드가 있음(placeholder 위치 + `encode_images_or_videos(...).shape[1]`) → 우리 구현은 이 동적 방식을 채택

---

## 4. MAD ↔ AVCD 불일치 사항 목록 (통일 필요, 서버 OURS 세팅으로 확정)

| 항목 | MAD | AVCD | 우리 임시 기본값(MAD repo 세팅) |
|---|---|---|---|
| AVHBench split | QA.json 전체에서 AV* task 제외 (3,426) | AVH_val(205) / AVH_test(5,302), AV Matching 포함 | MAD 방식 · `UNKNOWN_pending_server` |
| AVHBench 프롬프트 suffix | 4가지 혼재 (§1.5 표) | `" Answer yes or no."` | `ANSWER_QUERY_PROMPT` · `UNKNOWN_pending_server` |
| CMM 지원 | 있음 (category 필터, default는 over-reliance만) | 없음 | category=`UNKNOWN_pending_server` |
| 채점기 | 정규식 normalize (yes 우선), AV Captioning/Matching은 Overall 제외 | substring 매칭 | MAD `score.py` 이식 · 추후 OURS 채점기 교체 |
| attention 구현 | flash/sdpa 혼재 | 수동 fp32 eager (필수) | **eager 통일 (확정)** |
| max_new_tokens | AVHBench 1 / CMM 256~512 | 루프 상한 2048 (yes/no는 사실상 첫 토큰) | `UNKNOWN_pending_server` (임시: AVHBench 1, CMM 16) |
| 디코딩 | greedy | greedy | greedy temp=0 (확정) |
| 하이퍼파라미터 | γ=2.5 (sh 기준) | α=2.5(AVH), P=50%, τ=0.6, β=0.2(코드) | γ=2.5, α=2.5/CMM은 그리드, β는 §2.5 참고 |
| Qwen 모델 크기 | sh 예시 일부 3B (`batch_avhbench.sh`, `batch_cmm*.sh`) | — | **7B 고정** (blueprint) |

## 5. 구현에 반영할 리스크/결정 사항 요약

1. **VCD-ext 의미 이중성 → 잠정 정리**: MAD 논문 Eq.10 원문 확인 결과 ˜(perturbation)=modality 제거이며 코드 `mm_default_cd`와 일치 → **modality_ablation 잠정 채택** (`docs/paper_settings.md` §1·§5), 서버 OURS 확인 후 최종. α 값 미정({0.5, 2.5} 게이트 대조)
2. **AVCD 재구현 갭 2건** (MAD Table 1의 AVCD 수치는 MAD 저자 비공개 재구현 — repo에 코드 없음, `paper_settings.md` §6):
   ① AVCD 디코딩을 CMM 평가 파이프라인(MAD식 하네스)에 이식 ② **Qwen2.5-Omni용 신규 포팅** — span을 token id 기반 boolean mask로 동적 계산(interleave 대응), attention은 `attn_implementation="eager"` + `output_attentions=True`로 접근, "마지막 layer 제외"는 `num_layers−1`로 일반화. 실패 시 해당 칸 "공식 코드 미지원" 표기(blueprint §9-2)
3. **AVCD 재현 충실도**: β=0.2·`-1e-4` fill·threshold의 layer 누적 방식 등 코드 특이점을 그대로 재현하는 `faithful` 모드를 기본으로 하고, 논문 서술 기준(β=0.1, −inf) 대안을 yaml 옵션으로
4. **MAD 가중 수식**은 지시서 요약("γ·w 가중합")이 아닌 코드 수식(§1.4)으로 구현·테스트
5. AVCD 공식 short-answer 경로는 패치된 transformers에 의존(레포 부재) → 우리는 수동 디코딩 루프로 구현하며, AVHBench에서 AVCD 논문 수치와의 대조는 `generate_long` 경로 수식 기준
