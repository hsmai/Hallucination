# CLAUDE.md — Omni-Steering 비교군 실험 작업 규칙

> 출처: `claude_code_master_prompt.md`의 "제약"과 "작업 원칙" 섹션 발췌.
> 전체 계획은 `blueprint.md`, 상세 지시는 `claude_code_master_prompt.md` 참조. 충돌 시 blueprint 우선.

## 제약

1. **로컬 = macOS(Apple Silicon), GPU 없음.** 7B 모델 로드/forward 금지. CUDA 전용 패키지(flash-attn 등) 설치 시도 금지. 개발·검증은 CPU + 소형 더미 텐서 + MockModel dry-run으로만 한다.
2. **서버 계정 발급 대기 중.** 서버(A100 80GB×1)에는 이미 다음이 존재한다:
   - conda 환경 2종: `/home3/t202401082/.conda/envs/qwen-omni`, `/home3/t202401082/.conda/envs/videollama2` → 발급 후 복제해서 사용
   - OURS 코드+데이터셋: `/home3/t202401082/omni-steering` → 발급 후 직접 참조
   따라서 **로컬에서 conda env 구축, 모델 가중치 다운로드, 미디어 데이터 다운로드를 하지 않는다.** 서버 작업은 runbook 문서로만 준비한다. 벤치마크 QA json(경량)만 로컬로 받는다.
3. **세팅 미확정 항목 존재.** 프롬프트 템플릿, AVHBench split, CMM 구성, 채점기는 서버의 OURS 코드에서 확정된다. **어떤 세팅도 하드코딩하지 말고 전부 `configs/unified_settings.yaml`에서 읽는다.** 미확정 값은 `UNKNOWN_pending_server` + 임시 기본값(MAD repo 세팅)으로 둔다.
4. 모든 산출물을 커밋한다(서버에서 git clone 하나로 이전). 대용량은 .gitignore.

## 신규 코드 민감 구역 (2026-07-15 사용자 지시)

이 프로젝트의 코드는 두 등급으로 나뉘며 검증 강도를 달리한다:

- **이식 구역** (MAD/AVCD 오픈소스를 거의 무변경으로 사용): base, MAD 디코딩, AVCD×VideoLLaMA2(공식 fork 백엔드). 원본과의 diff가 곧 검증.
- **신규 구역 ⚠** (오픈소스가 없어 우리가 새로 작성 — 선배 지시상 반드시 동작해야 하며 "미지원 표기" fallback 사용 불가):
  1. **AVCD × Qwen2.5-Omni 포팅** (`src/adapters/qwen_omni.py`의 avcd_*, `src/adapters/attn_patch.py`, `src/adapters/common.py`)
  2. **AVCD × CMM 이식** (방법 플러그인화 — `src/methods/avcd.py`)
  3. VCD-extended (특히 Qwen용 — MAD repo에 dead code만 존재)
  신규 구역은: 수식 단위 테스트 + 로컬 통합 테스트 + 서버 스모크 + 게이트 수치 대조의 4단 검증을 전부 거친다.
  수정이 생기면 반드시 대응 테스트를 갱신하고, internals 로깅으로 중간값(dominance, weight, skip 비율)을 남겨 수치 이상 시 역추적 가능하게 유지한다.

## 작업 원칙

- Phase 완료마다 git commit(무엇을 검증했는지 메시지에 명시)하고, 사용자에게 (a)산출물 목록 (b)발견한 이슈 (c)다음 Phase 진행 여부를 보고한 뒤 진행한다.
- 논문 수치·하이퍼파라미터를 기억으로 단정하지 않는다. 지시 문서의 요약과 clone한 실제 코드가 근거이며, 둘이 충돌하면 코드를 우선하되 반드시 보고한다.
- 외부 repo 코드는 수정하지 않고 `third_party/`에 원본 유지, 우리 코드는 `src/`에 작성한다.
- 스크립트는 재실행 안전(idempotent)하게, 실패 시 명확한 에러 메시지를 낸다.
