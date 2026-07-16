#!/bin/bash
# PBS 배치 job 템플릿 — 우리 실행 스크립트를 페이로드로 감싼다.
#
# 사용 (repo 루트에서):
#   qsub -v PAYLOAD="scripts/run_smoke.sh"        scripts/pbs_job.sh
#   qsub -v PAYLOAD="scripts/run_gate.sh 200"     scripts/pbs_job.sh
#   qsub -v PAYLOAD="scripts/run_full.sh"         scripts/pbs_job.sh
#   상태: qstat / 종료: qdel <jobid> / 자원: pbsnodes -ajS
#
# ⚠ 제출 전 확인:
#   1) -N 이름을 연구실 규칙 "GPU수_CPU수_아이디_작업명"으로 수정 (아이디 부분)
#   2) -q 큐 이름이 pleiades3인지 확인: qstat -Q 로 실제 큐 목록 대조
#   3) GPU 사용 신청이 승인된 기간 내인지 (runbook §S0-2)
#
# 로그: 러너가 logs/*.log 를 직접 쓰므로 PBS 스풀과 무관하게 tail -f 가능.
#       PBS 자체 stdout/err은 잡 종료 후 <jobname>.o<jobid> 로 생성됨.

#PBS -l select=1:ncpus=8:ngpus=1
#PBS -N 1_8_CHANGEME_omnisteer
#PBS -q pleiades3
#PBS -r n
#PBS -j oe

cd "$PBS_O_WORKDIR"

# conda 초기화 (로그인 셸이 아니므로 명시적으로)
source ~/.bashrc 2>/dev/null || true
if ! command -v conda >/dev/null; then
  for c in /opt/anaconda3/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && source "$c" && break
  done
fi
command -v conda >/dev/null || { echo "ERROR: conda를 찾지 못함 — 경로를 이 스크립트에 추가"; exit 1; }

echo "[pbs_job] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES payload=${PAYLOAD:?PAYLOAD 미지정}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash $PAYLOAD
