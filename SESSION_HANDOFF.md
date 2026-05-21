# Session Handoff — 2026-05-22 (최신)

## 프로젝트 개요
- **목표**: Spot Micro 12-DOF 취미 사족보행 로봇 Isaac Lab + RSL-RL PPO 학습 → 실제 로봇 배포
- **Isaac Lab 환경**: DirectRLEnv, 4096 parallel envs, RSL-RL PPO
- **URDF**: `/home/wodnr/Downloads/spot_micro_light.urdf` (2.5 kg 경량 버전)
- **링크 길이**: L1(대퇴골)=0.1075m, L2(종아리)=0.130m
- **Python 환경**: `/home/wodnr/miniconda3/envs/env_isaaclab/bin/python`
- **작업 디렉토리**: `/home/wodnr/quadrupedal_bot/quadrupedal_bot`

---

## 커리큘럼 파이프라인 완료 ✅

### 전체 흐름
```
Stage 1 (Stance) → Stage 2 (TrotInplace) → Stage 3 (WalkFwd)
→ Stage 4 (WalkAllDir) → Stage 5 (InplaceRot)
```

---

## 완료된 체크포인트 (2026-05-22)

| Stage | 실험 이름 | Run 폴더 | 최종 체크포인트 | 주요 지표 |
|-------|-----------|----------|----------------|-----------|
| Stage 1: Stance | `spot_micro_stance_v3c` | `2026-05-22_02-02-36` | `model_1999.pt` | h=0.17m, term=0% |
| Stage 2: TrotInplace | `spot_micro_trot_inplace` | `2026-05-22_02-45-19` | `model_1999.pt` | gait>7, term=0% |
| Stage 3: WalkFwd | `spot_micro_walk_fwd` | `2026-05-22_03-29-33` | `model_2999.pt` | vel=0.4m/s, h=0.17m |
| Stage 4: WalkAllDir | `spot_micro_walk_alldir` | `2026-05-22_04-56-31` | `model_2999.pt` | h=0.170m, ang=0.47, gait=11.7 ✅ |
| Stage 5: InplaceRot | `spot_micro_inplace_rot` | `2026-05-22_06-02-45` | `model_1999.pt` | h=0.170m, ang=0.23, gait=11.5 ✅ |

---

## Stage별 상세 결과

### Stage 1: Stance ✅
- **Config**: `QuadrupedalBotStanceCfg` / `PPORunnerCfgStage1`
- **결과**: body_height=0.17m, term=0%, 안정적 기립

### Stage 2: TrotInplace ✅
- **Config**: `QuadrupedalBotTrotInplaceCfg` / `PPORunnerCfgStage3`
- **전이**: Stage 1 → TrotInplace
- **결과**: gait 시작, 제자리 트롯 학습

### Stage 3: WalkFwd ✅
- **Config**: `QuadrupedalBotWalkFwdCfg` / `PPORunnerCfgStage4`
- **전이**: TrotInplace → 전진 보행
- **cmd**: lin_vel_x: (0.1, 0.4)
- **결과**: vel=0.4m/s 달성, term=0%

### Stage 4: WalkAllDir ✅ (핵심 성과)
- **Config**: `QuadrupedalBotWalkAllDirCfg` / `PPORunnerCfgStage5`
- **전이**: WalkFwd → 전방향 보행
- **cmd**: lin_vel_x: (-0.3,0.4), lin_vel_y: (-0.2,0.2), ang_vel_z: (-1.0,1.0)
- **핵심 수정**: `rew_scale_body_height=30.0` (Stage3 낮은 높이 로컬옵티멈 탈출)
- **termination_height=0.135m** 하한선 추가
- **최종 지표**: h=0.170m, term=0%, gait=11.7, ang_vel_z=0.47 rad/s

### Stage 5: InplaceRot ✅
- **Config**: `QuadrupedalBotInplaceRotCfg` / `PPORunnerCfgStage6`
- **전이**: WalkAllDir → 제자리 회전 특화
- **cmd**: lin_vel_x: (-0.1,0.2), lin_vel_y: (-0.1,0.1), ang_vel_z: (-1.5,1.5)
- **회전 보상**: rew_scale_ang_vel=3.0, rew_scale_yaw_tracking=5.0
- **최종 지표**: h=0.170m, term=0%, gait=11.5, ang_vel_z=0.23 rad/s

---

## 핵심 문제 해결 이력

### 1. Stage 3 낮은 높이 로컬옵티멈
**문제**: Stage 3 (WalkFwd) 후 body_height=0.115m로 수렴 (목표 0.17m)
**원인**: WalkFwdCfg가 rew_scale_body_height를 설정하지 않아 기본값 3.0 상속 → gait 보상(~12)에 압도됨
**해결**: WalkAllDirCfg에서 rew_scale_body_height=30.0, termination_height=0.135m 추가

### 2. 사용 가능한 body_height reward 구조
```python
# exp(-|h - target| / 0.05) × scale  (scale > 0일 때 보상)
# 목표 h=0.17m, scale=30.0이면 gait(~12)보다 강력
```

---

## 현재 파일 구조

### 환경 설정 파일
```
source/quadrupedal_bot/quadrupedal_bot/tasks/direct/quadrupedal_bot/
├── quadrupedal_bot_stance_cfg.py     # Stage 1
├── quadrupedal_bot_trot_inplace_cfg.py  # Stage 2
├── quadrupedal_bot_walk_cfg.py       # Stage 3~5
│   ├── QuadrupedalBotWalkFwdCfg      # Stage 3
│   ├── QuadrupedalBotWalkAllDirCfg   # Stage 4
│   └── QuadrupedalBotInplaceRotCfg   # Stage 5
└── agents/rsl_rl_ppo_cfg_stage3.py   # Stage 3~5 PPO 러너
    ├── PPORunnerCfgStage4 (WalkFwd)
    ├── PPORunnerCfgStage5 (WalkAllDir)
    └── PPORunnerCfgStage6 (InplaceRot)
```

### obs_space: 52차원
- 기본 관절/속도/명령 + heading error

---

## 학습 실행 명령

```bash
cd /home/wodnr/quadrupedal_bot/quadrupedal_bot

# Stage 5 (InplaceRot) 재학습/이어서
/home/wodnr/miniconda3/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/train.py \
  --task Template-Quadrupedal-Bot-InplaceRot-v0 \
  --num_envs 4096 --headless

# 시각화 (최신 Stage 5 체크포인트)
/home/wodnr/miniconda3/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-InplaceRot-v0 \
  --num_envs 4

# 전방향 보행 시각화 (Stage 4)
/home/wodnr/miniconda3/envs/env_isaaclab/bin/python \
  scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-WalkAllDir-v0 \
  --num_envs 4
```

---

## 다음 가능한 작업

1. **시각화 확인**: Stage 4 WalkAllDir 또는 Stage 5 InplaceRot play.py로 실제 보행 확인
2. **실제 로봇 배포**: model_1999.pt / model_2999.pt → 실물 Spot Micro 전송
3. **Stage 5 회전 개선**: ang_vel_z=0.23 rad/s를 더 높이려면 rew_scale_ang_vel 추가 증가 또는 이어서 학습
4. **역방향/측방향 보행 품질 확인**: Stage 4 WalkAllDir에서 후진/좌우 실제 달성 여부 확인

---

## 환경 주의사항

- **Python 실행**: `python -m isaaclab` 아님! 직접 실행:
  `/home/wodnr/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py`
- **CUDA OOM 방지**: 새 훈련 전 `nvidia-smi`로 이전 프로세스 확인
- **체크포인트 저장 위치**: `logs/rsl_rl/<experiment_name>/<timestamp>/model_XXXX.pt`
- **저장 간격**: save_interval=200 iterations
