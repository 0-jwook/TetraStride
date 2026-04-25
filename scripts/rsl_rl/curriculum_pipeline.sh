#!/bin/bash
# 커리큘럼 자동 파이프라인: Stage1 → Stage2 → Stage3
# 각 스테이지 완료 시 체크포인트 심링크 연결 후 다음 스테이지 자동 진행

set -e

PYTHON="/home/wodnr/miniconda3/envs/env_isaaclab/bin/python"
WORKDIR="/home/wodnr/quadrupedal_bot/quadrupedal_bot"
LOG_BASE="$WORKDIR/logs/rsl_rl"
SCRIPT_DIR="$WORKDIR/scripts/rsl_rl"

cd "$WORKDIR"

log() {
    echo "[PIPELINE $(date '+%H:%M:%S')] $1"
}

# ─────────────────────────────────────────────
# STAGE 1: 서기 (Stance)
# ─────────────────────────────────────────────
log "========== STAGE 1 시작: Stance (5000 iter) =========="
$PYTHON "$SCRIPT_DIR/train.py" \
    --task Template-Quadrupedal-Bot-Stance-v0 \
    --num_envs 4096 --headless
log "========== STAGE 1 완료 =========="

# 최신 Stage 1 run 폴더 찾기
STAGE1_RUN=$(ls -td "$LOG_BASE/spot_micro_stance/2026-"* 2>/dev/null | head -1)
log "Stage 1 체크포인트: $STAGE1_RUN"

# Stage 2 폴더에 심링크 생성
mkdir -p "$LOG_BASE/spot_micro_trot"
STAGE1_LINK="$LOG_BASE/spot_micro_trot/stage1_stance"
ln -sfn "$STAGE1_RUN" "$STAGE1_LINK"
log "Stage 2용 심링크 생성: $STAGE1_LINK → $STAGE1_RUN"

# ─────────────────────────────────────────────
# STAGE 2: 트롯 보행 (Trot)
# ─────────────────────────────────────────────
log "========== STAGE 2 시작: Trot (8000 iter) =========="
$PYTHON "$SCRIPT_DIR/train.py" \
    --task Template-Quadrupedal-Bot-Trot-v0 \
    --num_envs 4096 --headless \
    --resume --load_run stage1_stance
log "========== STAGE 2 완료 =========="

# 최신 Stage 2 run 폴더 찾기
STAGE2_RUN=$(ls -td "$LOG_BASE/spot_micro_trot/2026-"* 2>/dev/null | head -1)
log "Stage 2 체크포인트: $STAGE2_RUN"

# Stage 3 폴더에 심링크 생성
mkdir -p "$LOG_BASE/spot_micro_locomotion"
STAGE2_LINK="$LOG_BASE/spot_micro_locomotion/stage2_trot"
ln -sfn "$STAGE2_RUN" "$STAGE2_LINK"
log "Stage 3용 심링크 생성: $STAGE2_LINK → $STAGE2_RUN"

# ─────────────────────────────────────────────
# STAGE 3: 전속도 보행 (Full Locomotion)
# ─────────────────────────────────────────────
log "========== STAGE 3 시작: Full Locomotion (10000 iter) =========="
$PYTHON "$SCRIPT_DIR/train.py" \
    --task Template-Quadrupedal-Bot-Direct-v0 \
    --num_envs 4096 --headless \
    --resume --load_run stage2_trot
log "========== STAGE 3 완료 =========="

STAGE3_RUN=$(ls -td "$LOG_BASE/spot_micro_locomotion/2026-"* 2>/dev/null | head -1)
log "Stage 3 체크포인트: $STAGE3_RUN"

log "========== 전체 커리큘럼 파이프라인 완료 =========="
log "Stage1: $STAGE1_RUN"
log "Stage2: $STAGE2_RUN"
log "Stage3: $STAGE3_RUN"
