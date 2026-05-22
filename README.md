# TetraStride — Spot Micro Quadruped Locomotion (Isaac Lab RL)

Spot Micro 사족보행 로봇의 완전한 보행 정책을 Isaac Lab + RSL-RL PPO 커리큘럼 학습으로 구현하는 프로젝트.

---

## 로봇 사양

| 항목 | 값 |
|------|-----|
| 모델 | Spot Micro (경량 버전) |
| 무게 | 2.5 kg |
| DOF | 12 (shoulder × 4, leg × 4, foot × 4) |
| Thigh 길이 | 107.5 mm |
| Calf 길이 | 130 mm |
| 기립 목표 높이 | 0.17 m |
| 기본 관절 자세 | shoulder=0.0, leg=+0.83 rad, foot=−1.55 rad (역관절) |
| URDF | `spot_micro_light.urdf` |

---

## 환경 구성

- **시뮬레이터**: Isaac Lab (IsaacSim + PhysX GPU)
- **RL 알고리즘**: RSL-RL PPO
- **병렬 환경 수**: 4096
- **Physics dt**: 1/200 Hz, decimation=4 (policy 50 Hz)
- **Observation space**: 56 dim
  - lin_vel_b (3) + ang_vel (3) + gravity (3) + cmd (3)
  - joint_pos_dev (12) + joint_vel (12) + actions (12)
  - gait_phase sin/cos (2) + heading_err sin/cos (2) + per_foot_clock (4)
- **Action space**: 12 (joint position targets, relative to default)
- **지면 마찰**: static=0.8, dynamic=0.6 (현실적 수치)

---

## 커리큘럼 파이프라인

```
Stage 1: Stance (서있기)
  └─ 역관절 자세(foot=−1.55) 유지, 제자리 안정
  └─ task: Template-Quadrupedal-Bot-Stance-v0
  └─ cfg: quadrupedal_bot_stance_cfg.py

Stage 2: TrotInplace (제자리 트롯)
  └─ Stage 1 전이 → 제자리 대각선 trot
  └─ task: Template-Quadrupedal-Bot-TrotInplace-v0
  └─ cfg: quadrupedal_bot_trot_inplace_cfg.py

Stage 3: WalkFwd (전진 보행)
  └─ Stage 2 전이 → 전진 보행 (0.1~0.4 m/s)
  └─ task: Template-Quadrupedal-Bot-WalkFwd-v0
  └─ cfg: quadrupedal_bot_walk_cfg.py (WalkFwdCfg)

Stage 4: WalkAllDir (전방향 보행)
  └─ Stage 3 전이 → 전후진+좌우+회전
  └─ task: Template-Quadrupedal-Bot-WalkAllDir-v0
  └─ cfg: quadrupedal_bot_walk_cfg.py (WalkAllDirCfg)

Stage 5: InplaceRot (제자리 회전)
  └─ Stage 4 전이 → 제자리 yaw 회전 특화
  └─ task: Template-Quadrupedal-Bot-InplaceRot-v0
  └─ cfg: quadrupedal_bot_walk_cfg.py (InplaceRotCfg)
```

---

## 핵심 설계 결정

### 역관절 자세 (foot=−1.55 rad)
- 기존 `foot=−0.83`은 calf 수직 → CoM이 발 뒤쪽 → 마찰 의존 불안정
- `foot=−1.55`는 calf 41° 전방 경사 → 발이 CoM 직하방 → 물리적으로 안정
- 지면 마찰을 현실적 수치(0.8/0.6)로 낮춰도 서있을 수 있는 자세

### 높이 유지 전략
- `rew_scale_body_height = 30.0` (WalkAllDir): gait 보상을 압도하여 낮은 자세 로컬옵티멈 방지
- `termination_height = 0.135m`: 지나치게 낮은 자세 에피소드 강제 종료

---

## 설치

```bash
# Isaac Lab 설치 후
conda activate env_isaaclab
cd /home/wodnr/quadrupedal_bot/quadrupedal_bot
pip install -e source/quadrupedal_bot
```

---

## 학습 실행

```bash
cd /home/wodnr/quadrupedal_bot/quadrupedal_bot
PYTHON=/home/wodnr/miniconda3/envs/env_isaaclab/bin/python

# Stage 1: Stance
$PYTHON scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4096 --headless

# Stage 2: TrotInplace
$PYTHON scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-TrotInplace-v0 --num_envs 4096 --headless

# Stage 3: WalkFwd
$PYTHON scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-WalkFwd-v0 --num_envs 4096 --headless

# Stage 4: WalkAllDir
$PYTHON scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-WalkAllDir-v0 --num_envs 4096 --headless

# Stage 5: InplaceRot
$PYTHON scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-InplaceRot-v0 --num_envs 4096 --headless
```

## 시각화

```bash
# 체크포인트 절대 경로 지정
$PYTHON scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-WalkAllDir-v0 \
  --num_envs 4 \
  --checkpoint /home/wodnr/quadrupedal_bot/quadrupedal_bot/logs/rsl_rl/spot_micro_walk_alldir/2026-05-22_04-56-31/model_2999.pt
```

---

## 주요 파일 구조

```
source/quadrupedal_bot/quadrupedal_bot/tasks/direct/quadrupedal_bot/
├── spot_micro_cfg.py                   # 로봇 물리/관절 설정
├── quadrupedal_bot_env.py              # 환경 구현 (보상 함수)
├── quadrupedal_bot_env_cfg.py          # 기본 환경 파라미터
├── quadrupedal_bot_stance_cfg.py       # Stage 1: 서있기
├── quadrupedal_bot_trot_inplace_cfg.py # Stage 2: 제자리 트롯
├── quadrupedal_bot_walk_cfg.py         # Stage 3~5: 보행/회전
│   ├── QuadrupedalBotWalkFwdCfg
│   ├── QuadrupedalBotWalkAllDirCfg
│   └── QuadrupedalBotInplaceRotCfg
└── agents/
    ├── rsl_rl_ppo_cfg_stage1.py        # Stage 1 PPO
    ├── rsl_rl_ppo_cfg_stage2.py        # Stage 2 PPO (TrotInplace seed)
    └── rsl_rl_ppo_cfg_stage3.py        # Stage 3~5 PPO
```

---

## 완료된 체크포인트 (2026-05-22, 구버전 foot=−0.83)

> ⚠️ 아래 체크포인트는 foot=−0.83 기준. 현재 재학습(foot=−1.55) 진행 중.

| Stage | 실험 이름 | Run | 체크포인트 | 주요 지표 |
|-------|-----------|-----|-----------|-----------|
| Stance | spot_micro_stance_rev | 2026-05-22_15-03-36 | model_2999.pt | h=0.167m |
| TrotInplace | spot_micro_trot_inplace | 2026-05-22_02-45-19 | model_1999.pt | gait>7 |
| WalkFwd | spot_micro_walk_fwd | 2026-05-22_03-29-33 | model_2999.pt | vel=0.4 m/s |
| WalkAllDir | spot_micro_walk_alldir | 2026-05-22_04-56-31 | model_2999.pt | h=0.170m, ang=0.47 rad/s ✅ |
| InplaceRot | spot_micro_inplace_rot | 2026-05-22_06-02-45 | model_1999.pt | h=0.170m, ang=0.23 rad/s ✅ |

---

## 보상 구조 요약

### Stage 1 (Stance)
| 항목 | 값 | 역할 |
|------|----|------|
| `rew_scale_body_height` | +5.0 | 역관절 높이 유지 |
| `rew_scale_gravity` | −5.0 | 기울어짐 패널티 |
| `rew_scale_upright` | +2.0 | 직립 보상 |
| `rew_scale_non_foot_contact` | −2.0 | 무릎/배 접지 방지 |
| `rew_scale_lin_vel_xy` | −2.0 | 수평 이동 방지 |
| `rew_scale_stand_still` | −1.0 | 제자리 유지 |

### Stage 4 (WalkAllDir)
| 항목 | 값 | 역할 |
|------|----|------|
| `rew_scale_body_height` | +30.0 | 높이 로컬옵티멈 탈출 |
| `rew_scale_gait` | +12.0 | trot 패턴 |
| `rew_scale_air_time` | +10.0 | 발 들기 |
| `termination_height` | 0.135 m | 낮은 자세 에피소드 종료 |

---

## 해결한 주요 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| 역관절 미형성 | foot 기본값 −0.83 → calf 수직, CoM 앞쏠림 | foot=−1.55으로 변경 |
| 서있기 불안정 | 마찰 1.0이 물리적 불균형 숨김 | 마찰 0.8/0.6으로 현실화 |
| Stage 3 낮은 자세 | body_height 보상(3.0)이 gait(12)에 압도 | body_height=30.0 + termination 추가 |
| 제자리 sliding | stand_still/lin_vel_xy 패널티 약함 | 각각 −1.0/−2.0으로 강화 |

---

## 환경 주의사항

- **Python 실행**: `python -m isaaclab` ❌ → 직접 실행 ✅
  ```bash
  /home/wodnr/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py ...
  ```
- **CUDA OOM**: 학습 전 `nvidia-smi`로 이전 프로세스 확인 후 `kill -9`
- **체크포인트**: `logs/rsl_rl/<experiment_name>/<timestamp>/model_XXXX.pt`
- **저장 간격**: `save_interval=200` iterations
