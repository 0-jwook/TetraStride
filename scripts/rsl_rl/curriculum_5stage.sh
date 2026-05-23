#!/bin/bash
# 5단계 커리큘럼 파이프라인: Stance → TrotInplace → WalkFwd → WalkAllDir → InplaceRot
# 각 스테이지 완료 후 자동으로 다음 스테이지 진행 (시각화 없음)

set -e

PYTHON="/home/wodnr/miniconda3/envs/env_isaaclab/bin/python"
WORKDIR="/home/wodnr/quadrupedal_bot/quadrupedal_bot"
LOG_BASE="$WORKDIR/logs/rsl_rl"
SCRIPT="$WORKDIR/scripts/rsl_rl/train.py"

cd "$WORKDIR"

log() { echo "[PIPELINE $(date '+%H:%M:%S')] $1"; }

find_latest_model() {
    local run_dir="$1"
    ls "$run_dir"/model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1 | xargs basename 2>/dev/null
}

# ─── STAGE 1: Stance ─────────────────────────────────────────────────────────
log "========== STAGE 1 시작: Stance =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-Stance-v0 \
    --num_envs 4096 --headless
log "========== STAGE 1 완료 =========="

S1_RUN=$(ls -td "$LOG_BASE/spot_micro_stance_v4/2026-"* 2>/dev/null | head -1)
S1_TS=$(basename "$S1_RUN")
S1_CKPT=$(find_latest_model "$S1_RUN")
log "Stage 1 체크포인트: $S1_TS / $S1_CKPT"

# ─── STAGE 2: TrotInplace ────────────────────────────────────────────────────
log "========== STAGE 2 시작: TrotInplace (← Stance $S1_TS) =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-TrotInplace-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "$S1_TS" --checkpoint "$S1_CKPT"
log "========== STAGE 2 완료 =========="

S2_RUN=$(ls -td "$LOG_BASE/spot_micro_trot_inplace_kp30/2026-"* 2>/dev/null | head -1)
S2_TS=$(basename "$S2_RUN")
S2_CKPT=$(find_latest_model "$S2_RUN")
log "Stage 2 체크포인트: $S2_TS / $S2_CKPT"

# ─── STAGE 3: WalkFwd ────────────────────────────────────────────────────────
log "========== STAGE 3 시작: WalkFwd (← TrotInplace $S2_TS) =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-WalkFwd-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "$S2_TS" --checkpoint "$S2_CKPT"
log "========== STAGE 3 완료 =========="

S3_RUN=$(ls -td "$LOG_BASE/spot_micro_walk_fwd_kp30/2026-"* 2>/dev/null | head -1)
S3_TS=$(basename "$S3_RUN")
S3_CKPT=$(find_latest_model "$S3_RUN")
log "Stage 3 체크포인트: $S3_TS / $S3_CKPT"

# ─── STAGE 4: WalkAllDir ─────────────────────────────────────────────────────
log "========== STAGE 4 시작: WalkAllDir (← WalkFwd $S3_TS) =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-WalkAllDir-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "$S3_TS" --checkpoint "$S3_CKPT"
log "========== STAGE 4 완료 =========="

S4_RUN=$(ls -td "$LOG_BASE/spot_micro_walk_alldir_kp30/2026-"* 2>/dev/null | head -1)
S4_TS=$(basename "$S4_RUN")
S4_CKPT=$(find_latest_model "$S4_RUN")
log "Stage 4 체크포인트: $S4_TS / $S4_CKPT"

# ─── STAGE 5: InplaceRot ─────────────────────────────────────────────────────
log "========== STAGE 5 시작: InplaceRot (← WalkAllDir $S4_TS) =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-InplaceRot-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "$S4_TS" --checkpoint "$S4_CKPT"
log "========== STAGE 5 완료 =========="

S5_RUN=$(ls -td "$LOG_BASE/spot_micro_inplace_rot_kp30/2026-"* 2>/dev/null | head -1)
S5_CKPT=$(find_latest_model "$S5_RUN")
log "Stage 5 체크포인트: $S5_RUN/$S5_CKPT"

log "=========================================="
log "5단계 커리큘럼 전체 완료"
log "  S1 Stance:     $LOG_BASE/spot_micro_stance_v4/$S1_TS/$S1_CKPT"
log "  S2 TrotInplace:$LOG_BASE/spot_micro_trot_inplace_kp30/$S2_TS/$S2_CKPT"
log "  S3 WalkFwd:    $LOG_BASE/spot_micro_walk_fwd_kp30/$S3_TS/$S3_CKPT"
log "  S4 WalkAllDir: $LOG_BASE/spot_micro_walk_alldir_kp30/$S4_TS/$S4_CKPT"
log "  S5 InplaceRot: $S5_RUN/$S5_CKPT"
log "=========================================="
