# 데이터 리포트 (Phase L2)

> QA annotation만 로컬 확보 (제약 2: 미디어 다운로드 금지). 로더: `src/data.py`, 설정: `configs/unified_settings.yaml`.

## 1. 확보 파일과 출처

| 파일 | 출처 | 크기 | 내용 |
|---|---|---|---|
| `data/qa/avhbench_qa.json` | `third_party/AVCD/json/QA.json` 복사 (원출처 kaist-ami/AVHBench) | 1.1MB | 전체 6,408문항 |
| `data/qa/avhbench_val.json` | `third_party/AVCD/json/AVH_val.json` | 36KB | AVCD 논문 validation 205문항 |
| `data/qa/avhbench_test.json` | `third_party/AVCD/json/AVH_test.json` | 696KB | AVCD 논문 test 5,302문항 |
| `data/qa/cmm_qa.json` | [HF DAMO-NLP-SG/CMM](https://huggingface.co/datasets/DAMO-NLP-SG/CMM) `all_data_final_reorg.json` | 1.2MB | 2,400문항 — **MAD repo 스크립트가 기대하는 파일명·스키마와 동일** |

- 다운로드 실패 항목: 없음
- 미디어: CMM `reorg_raw_files.zip`(4.7GB)·AVHBench 비디오는 **의도적으로 미다운로드** (서버에 존재 가정, yaml 프리픽스로 조립)

## 2. AVHBench

스키마: `{video_id, task, text, label}` (label = Yes/No)

| task | QA.json 전체 | 채점 대상(non-AV) |
|---|---|---|
| Video-driven Audio Hallucination | 2,290 | 2,290 |
| Audio-driven Video Hallucination | 1,136 | 1,136 |
| AV Matching | 1,876 | — (MAD 프로토콜에서 제외) |
| AV Captioning | 1,106 | — (〃) |
| 계 | 6,408 | 3,426 → **dedup 후 3,419** |

- Yes/No 균형: non-AV 3,426 기준 1,713 / 1,713
- ⚠ **(video_id, text) 중복 7쌍 발견** — 그중 2쌍은 라벨까지 상충(`01444` Yes/No, `00332` Yes/No — 데이터 노이즈).
  MAD `score.py`가 first-occurrence dedup을 하는 이유. **로더에서 동일 규칙(첫 등장 유지)으로 dedup** → 유효 3,419.
  avcd_test/full split도 동일 7건 제거(5,302→5,295, 6,408→6,401)
- split 옵션 4종 로더 검증 완료: `full_qa_excluding_av_tasks`(3,419) / `full_qa_all_tasks`(6,401) / `avcd_val`(205) / `avcd_test`(5,295)
- 미디어 경로 규약 (MAD repo): `{prefix}/videos/{video_id}.mp4`, `{prefix}/audios/{video_id}.wav`

## 3. CMM

스키마: `{category, sub_category, modality, granularity, correlation_type, video_path, audio_path, question, answer}` (answer = yes/no 소문자)

| category | sub_category | n | modality | 비고 |
|---|---|---|---|---|
| over-reliance_unimodal_priors | overrely_visual_ignore_audio | 400 | visual+audio | = **Visual Dominance** |
| 〃 | overrely_audio_ignore_visual | 400 | visual+audio | = **Audio Dominance** |
| 〃 | overrely_language_ignore_visual | 400 | visual (audio 없음) | = **Language Dominance** |
| inter-modality_spurious_correlation | visual-language | 400 | visual (audio 없음) | 논문 표 미사용 |
| 〃 | audio-language | 400 | audio (video 없음) | 〃 |
| 〃 | visual-audio-language | 400 | visual+audio | 〃 |

- **L1 가설 데이터로 확인**: MAD 논문 Table 1의 Visual/Audio/Language Dom = over-reliance 카테고리의 3개 sub_category (각 400, 계 1,200). Overall Acc는 이 1,200 기준으로 추정 (400×3 균등이므로 세 지표 평균 = Overall). → yaml temp_default `over-reliance_unimodal_priors` 유지, **최종 확정은 서버 OURS 코드에서**
- yes/no 균형: 전체 1,200/1,200, over-reliance 내에서도 각 sub_category 200/200
- granularity: event-level 1,400 / object-level 1,000
- ⚠ **video_path가 문자열 "None"인 항목 400건** (audio-language) — 로더에서 None으로 정규화, video_id는 audio stem 사용
- 키 유일성: (video_path, audio_path, question) 완전 유일 (2,400). **stem+question도 유일** → sample_id = `{stem}::{question}` 안전
- ⚠ Language Dominance(및 visual-language)는 **audio_path 없음** → 러너의 "av" 입력 구성 시 audio 부재 처리 필요 (L3에서 반영)

## 4. join 규약 (선배와 합의 필요 항목)

- `sample_id = "{video_id}::{question}"`
  - AVHBench: video_id = 원본 `video_id` ("00159"), question = 원본 `text` (suffix 미포함)
  - CMM: video_id = 미디어 파일 stem ("oxD5gs")
- question은 **프롬프트 suffix가 붙기 전 원본**을 사용한다 (suffix는 yaml 소관이라 변동 가능하므로 키에 넣지 않음)
- 중복 키는 양쪽 모두 first-occurrence 규칙 적용을 전제 (MAD score.py와 동일)

## 5. 로더 사용법

```python
from src.config import load_config
from src.data import load_benchmark

cfg = load_config()                      # configs/unified_settings.yaml
samples = load_benchmark(cfg, "avhbench")  # or "cmm"
# 서버에서: yaml의 paths.*_media_dir 확정 + skip_media_existence_check=false 로 전환
# OURS QA 파일로 교체 시: benchmarks.*.qa_json 경로만 변경
```

빠른 점검: `python -m src.data`
