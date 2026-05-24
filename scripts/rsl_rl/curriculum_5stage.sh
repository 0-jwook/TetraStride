#!/bin/bash
# 5단계 커리큘럼 파이프라인: Stance → TrotInplace → WalkFwd → WalkAllDir → InplaceRot
# RSL-RL은 load_run을 현재 experiment 디렉토리 내에서 찾으므로 심링크를 통해 전달

set -e

PYTHON="/home/wodnr/miniconda3/envs/env_isaaclab/bin/python"
WORKDIR="/home/wodnr/quadrupedal_bot/quadrupedal_bot"
LOG_BASE="$WORKDIR/logs/rsl_rl"
SCRIPT="$WORKDIR/scripts/rsl_rl/train.py"

cd "$WORKDIR"

log() { echo "[PIPELINE $(date '+%H:%M:%S')] $1"; }

find_latest_model() {
    local run_dir="$1"
    ls "$run_dir"/model_*.pt 2>/dev/null \
        | xargs -n1 basename \
        | sort -t_ -k2 -n \
        | tail -1
}

# 다음 스테이지 experiment 디렉토리에 심링크 생성
# RSL-RL이 "<next_exp>/<link_name>/<checkpoint>" 로 로드
make_link() {
    local src="$1"   # 실제 run 폴더 전체 경로
    local dst_dir="$2"  # 다음 스테이지 experiment 디렉토리
    local link_name="$3"  # 심링크 이름
    mkdir -p "$dst_dir"
    ln -sfn "$src" "$dst_dir/$link_name"
    log "심링크: $dst_dir/$link_name → $src"
}

# ─── STAGE 1: Stance ─────────────────────────────────────────────────────────
log "========== STAGE 1 시작: Stance =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-Stance-v0 \
    --num_envs 4096 --headless
log "========== STAGE 1 완료 =========="

S1_RUN=$(ls -td "$LOG_BASE/spot_micro_stance_v4/2026-"* 2>/dev/null | head -1)
S1_CKPT=$(find_latest_model "$S1_RUN")
log "Stage 1: $(basename $S1_RUN) / $S1_CKPT"

make_link "$S1_RUN" "$LOG_BASE/spot_micro_trot_inplace_kp30" "prev_stance"

# ─── STAGE 2: TrotInplace ────────────────────────────────────────────────────
log "========== STAGE 2 시작: TrotInplace =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-TrotInplace-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "prev_stance" --checkpoint "$S1_CKPT"
log "========== STAGE 2 완료 =========="

S2_RUN=$(ls -td "$LOG_BASE/spot_micro_trot_inplace_kp30/2026-"* 2>/dev/null | head -1)
S2_CKPT=$(find_latest_model "$S2_RUN")
log "Stage 2: $(basename $S2_RUN) / $S2_CKPT"

make_link "$S2_RUN" "$LOG_BASE/spot_micro_walk_fwd_kp30" "prev_trot_inplace"

# ─── STAGE 3: WalkFwd ────────────────────────────────────────────────────────
log "========== STAGE 3 시작: WalkFwd =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-WalkFwd-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "prev_trot_inplace" --checkpoint "$S2_CKPT"
log "========== STAGE 3 완료 =========="

S3_RUN=$(ls -td "$LOG_BASE/spot_micro_walk_fwd_kp30/2026-"* 2>/dev/null | head -1)
S3_CKPT=$(find_latest_model "$S3_RUN")
log "Stage 3: $(basename $S3_RUN) / $S3_CKPT"

make_link "$S3_RUN" "$LOG_BASE/spot_micro_walk_alldir_kp30" "prev_walk_fwd"

# ─── STAGE 4: WalkAllDir ─────────────────────────────────────────────────────
log "========== STAGE 4 시작: WalkAllDir =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-WalkAllDir-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "prev_walk_fwd" --checkpoint "$S3_CKPT"
log "========== STAGE 4 완료 =========="

S4_RUN=$(ls -td "$LOG_BASE/spot_micro_walk_alldir_kp30/2026-"* 2>/dev/null | head -1)
S4_CKPT=$(find_latest_model "$S4_RUN")
log "Stage 4: $(basename $S4_RUN) / $S4_CKPT"

make_link "$S4_RUN" "$LOG_BASE/spot_micro_inplace_rot_kp30" "prev_walk_alldir"

# ─── STAGE 5: InplaceRot ─────────────────────────────────────────────────────
log "========== STAGE 5 시작: InplaceRot =========="
$PYTHON "$SCRIPT" \
    --task Template-Quadrupedal-Bot-InplaceRot-v0 \
    --num_envs 4096 --headless \
    --resume --load_run "prev_walk_alldir" --checkpoint "$S4_CKPT"
log "========== STAGE 5 완료 =========="

S5_RUN=$(ls -td "$LOG_BASE/spot_micro_inplace_rot_kp30/2026-"* 2>/dev/null | head -1)
S5_CKPT=$(find_latest_model "$S5_RUN")

log "=========================================="
log "5단계 커리큘럼 전체 완료"
log "  S1 Stance:      $(basename $S1_RUN) / $S1_CKPT"
log "  S2 TrotInplace: $(basename $S2_RUN) / $S2_CKPT"
log "  S3 WalkFwd:     $(basename $S3_RUN) / $S3_CKPT"
log "  S4 WalkAllDir:  $(basename $S4_RUN) / $S4_CKPT"
log "  S5 InplaceRot:  $(basename $S5_RUN) / $S5_CKPT"
log "=========================================="
