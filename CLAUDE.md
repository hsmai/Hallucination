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

## 작업 원칙

- Phase 완료마다 git commit(무엇을 검증했는지 메시지에 명시)하고, 사용자에게 (a)산출물 목록 (b)발견한 이슈 (c)다음 Phase 진행 여부를 보고한 뒤 진행한다.
- 논문 수치·하이퍼파라미터를 기억으로 단정하지 않는다. 지시 문서의 요약과 clone한 실제 코드가 근거이며, 둘이 충돌하면 코드를 우선하되 반드시 보고한다.
- 외부 repo 코드는 수정하지 않고 `third_party/`에 원본 유지, 우리 코드는 `src/`에 작성한다.
- 스크립트는 재실행 안전(idempotent)하게, 실패 시 명확한 에러 메시지를 낸다.
