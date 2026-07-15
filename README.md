# Omni-Steering 비교군(Baseline) 실험

Omni-modal LLM의 cross-modal hallucination 완화 기법 **Omni-Steering(OURS, 선배 연구)** 논문을 위한
**비교군 실험 작업 공간**입니다. 기존 방법 4종을 동일 세팅으로 실행하여 정량 수치와 샘플별 출력을 확보합니다.

```
모델 2종     : VideoLLaMA2-AV,  Qwen2.5-Omni-7B
방법 4종     : Base, VCD-Extended, AVCD, MAD          (전부 training-free decoding)
벤치마크 2종 : CMM (Visual/Audio/Language Dominance),  AVHBench (V→A / A→V Hallucination)
              → 총 16 run
```

## 목표 (산출물)

1. **정량 — MAD 논문 Table 1을 우리 손으로 재생산**
   비교군 4종 × 2모델을 서버 동일 환경·동일 데이터에서 직접 실행 →
   CMM(Visual/Audio/Language Dom + Overall) · AVHBench(VdAH/AdVH + Overall) 수치 확보.
   OURS 수치와 함께 논문 메인 비교 테이블에 들어간다.
2. **정성 — 방법별 답변 비교 샘플** (MAD 논문 Fig.9-10 스타일)
   같은 입력에 대한 Base/VCD/AVCD/MAD의 raw 답변 + 비디오 프레임 + 장면 설명을 샘플당 폴더로 패키징.
   per-sample 결과는 `video_id + question` 키로 OURS와 join되어 "OURS만 정답인 샘플" 선별에 사용.

## 현재 상태

**로컬 준비(L1~L5) 완료 — 서버 계정 발급 대기 중.**
발급 후에는 [docs/server_runbook.md](docs/server_runbook.md)를 위에서부터 복붙 실행하면 된다.

| Phase | 내용 | 핵심 산출물 |
|---|---|---|
| **L1** 코드 정독·세팅 | MAD/AVCD/VideoLLaMA2 코드를 라인 단위로 분석, 논문 4건과 대조해 수식·충돌 확정. 모든 세팅을 yaml 하나로 외부화 (미확정 14개는 `UNKNOWN_pending_server`) | [docs/code_analysis.md](docs/code_analysis.md) · [docs/paper_settings.md](docs/paper_settings.md) · [configs/unified_settings.yaml](configs/unified_settings.yaml) |
| **L2** 데이터 | AVHBench(6,408)·CMM(2,400) QA json 확보(미디어 제외), 두 벤치마크를 공통 `Sample` 인터페이스로 읽는 로더. 중복 키 7쌍(라벨 상충 2쌍) 발견·처리 | [src/data.py](src/data.py) · [docs/data_report.md](docs/data_report.md) |
| **L3** 하네스 | 방법 플러그인 러너 + D2 스키마 JSONL + **중단-재개** + 채점(MAD 이식) + Table 1 형식 집계 + D3 마이닝. MockModel로 **16조합 전체 dry-run 통과** | [src/runner.py](src/runner.py) · [src/score.py](src/score.py) · [src/aggregate.py](src/aggregate.py) · [src/mining.py](src/mining.py) |
| **L4** 방법 구현 | 방법 4종 + 실모델 어댑터 2종. **AVCD 갭 2건 해소**: ① CMM 파이프라인 이식 ② Qwen2.5-Omni 신규 포팅(attention 패치 + 동적 span). 수식을 pytest 41건으로 논문·공식코드와 일치 고정 | [src/methods/](src/methods) · [src/adapters/](src/adapters) · [tests/](tests) |
| **L5** 서버 준비 | 발급 당일 복붙용 runbook(세팅 추출 체크리스트·게이트 기준·트러블슈팅) + 실행 스크립트 3종 | [docs/server_runbook.md](docs/server_runbook.md) · [scripts/](scripts) |

## 설계 원칙

- **세팅 하드코딩 금지** — 프롬프트/split/하이퍼파라미터/경로 전부 [configs/unified_settings.yaml](configs/unified_settings.yaml)에서만 읽는다.
  서버에서 OURS 코드를 확인해 UNKNOWN 14개를 확정하면 코드 수정 없이 전체 파이프라인이 정렬된다.
- **이식 구역 vs 신규 구역** ([CLAUDE.md](CLAUDE.md)) — 오픈소스를 거의 무변경으로 쓰는 부분(Base/MAD/AVCD×VideoLLaMA2)과
  달리, 오픈소스가 없어 새로 작성한 부분(**AVCD×Qwen 포팅, AVCD×CMM 이식, VCD-ext**)은
  4단 검증(수식 pytest → 로컬 통합 테스트 → 서버 스모크 → 게이트 수치 대조)을 강제한다.
- **재현성** — 모든 run은 JSONL + manifest(설정 해시·git 커밋) 기록, 중단돼도 재실행 시 이어서 처리.
- **게이트 우선** — 본 실행 전에 Base ≈ Ours(Base) 일치(AVH 77.4/76.8 등)를 관문으로 세팅 통일을 검증한다.

## 빠른 사용법

```bash
# 의존성 (모델 의존성은 서버 conda env가 제공)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 로컬 테스트용

# 참조 repo 3종 클론 (고정 커밋)
bash scripts/setup_third_party.sh

# 단위 테스트 (41건)
python -m pytest tests/ -q

# 로컬 dry-run: MockModel로 16조합 전체 배관 검증 (GPU 불필요)
bash scripts/run_dryrun_all.sh

# 단일 조합 실행 예 (서버, GPU)
python -m src.runner --model qwen2_5_omni_7b --method avcd --benchmark cmm
python -m src.score --jsonl "results/runs/cmm/*.jsonl"
python -m src.aggregate --results results/runs        # Table 1 형식 + 게이트 대조
```

## 저장소 구조

```
configs/unified_settings.yaml   # 모든 세팅의 단일 소스 (확정값 / UNKNOWN_pending_server 구분)
data/qa/                        # 벤치마크 QA json (경량 — 미디어는 서버에만)
src/
  config.py, data.py            # 설정·데이터 공통 로더
  runner.py                     # 실험 러너 (JSONL 로깅, 중단-재개, --set 오버라이드)
  score.py, aggregate.py        # 채점(OURS 채점기 교체 지점) · D1 테이블 생성
  mining.py                     # D3: MAD·AVCD 동시 오답 샘플 마이닝
  methods/                      # base / vcd_ext / mad / avcd 디코딩 플러그인
  adapters/                     # 실모델 어댑터 + AVCD attention 수학·패치 (신규 구역)
scripts/
  run_smoke.sh / run_gate.sh / run_full.sh   # 서버 실행 3단계
  package_samples.py            # 정성 샘플 패키징 (프레임 + 답변 + 장면 설명)
tests/                          # CPU 단위·통합 테스트 41건
docs/                           # 분석·데이터 리포트·서버 runbook
third_party/                    # MAD / AVCD / VideoLLaMA2 원본 (gitignore, 스크립트로 고정 클론)
```

## 참고 문서

- [blueprint.md](blueprint.md) — 전체 실험 계획 (최우선 기준 문서)
- [docs/paper_settings.md](docs/paper_settings.md) — MAD/AVCD/Omni-Steering 논문 발췌·게이트 목표치
- [docs/code_analysis.md](docs/code_analysis.md) — 참조 코드 3종 분석 (수식·span·채점 규칙 근거)
- [docs/server_runbook.md](docs/server_runbook.md) — 서버 발급 당일 실행 절차
