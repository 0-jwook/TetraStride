# TetraStride — Spot Micro Quadruped Locomotion (Isaac Lab RL)

Spot Micro 사족보행 로봇의 trot 보행을 Isaac Lab + RSL-RL PPO로 학습하는 프로젝트.

---

## 로봇 사양

| 항목 | 값 |
|------|-----|
| 모델 | Spot Micro |
| 무게 | 2.5 kg |
| DOF | 12 (shoulder × 4, thigh × 4, calf × 4) |
| Thigh 길이 | 107.5 mm |
| Calf 길이 | 130 mm |
| 기립 높이 | ~0.24 m |
| Foot 형태 | Sphere |

---

## 환경 구성

- **시뮬레이터**: Isaac Lab (IsaacSim + PhysX GPU)
- **RL 알고리즘**: RSL-RL PPO
- **병렬 환경 수**: 4096
- **Control frequency**: ~50 Hz (decimation 기반)
- **Observation space**: 52 dim
  - lin_vel_b (3) + ang_vel (3) + gravity (3) + cmd (3)
  - joint_pos (12) + joint_vel (12) + actions (12)
  - gait_phase sin/cos (2) + heading_err sin/cos (2)
- **Action space**: 12 (joint position targets, scale=0.35)

---

## 학습 구조

```
Stage 1: Standing
  → 기립 자세 안정화
  → 파일: quadrupedal_bot_env_cfg.py (stance)

Stage 2: Trot Locomotion (전이학습)
  → Stage 1 체크포인트에서 전이
  → 파일: quadrupedal_bot_trot_cfg.py
  → PPO 설정: agents/rsl_rl_ppo_cfg_stage2.py
```

---

## 설치

```bash
# Isaac Lab 설치 후
conda activate env_isaaclab
pip install -e source/quadrupedal_bot
```

---

## 학습 실행

모든 명령어는 `/home/wodnr/quadrupedal_bot/quadrupedal_bot/` 디렉토리에서 실행.

```bash
# Stage 2 Trot 학습 (headless GPU)
DISPLAY=:1 conda run -n env_isaaclab python scripts/rsl_rl/train.py \
  --task Template-Quadrupedal-Bot-Trot-v0 \
  --num_envs 4096 --headless

# 시각화 (run 이름 지정)
DISPLAY=:1 conda run -n env_isaaclab python scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-Trot-v0 \
  --num_envs 4 --load_run 2026-05-14_13-39-47

# 체크포인트 직접 지정 시 (절대 경로 필요)
DISPLAY=:1 conda run -n env_isaaclab python scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-Trot-v0 \
  --num_envs 4 \
  --checkpoint /home/wodnr/quadrupedal_bot/quadrupedal_bot/logs/rsl_rl/spot_micro_trot/<run_name>/model_4999.pt
```

---

## 주요 파일

```
source/quadrupedal_bot/quadrupedal_bot/tasks/direct/quadrupedal_bot/
├── quadrupedal_bot_env.py          # 환경 구현 (보상 함수 포함)
├── quadrupedal_bot_env_cfg.py      # Stage 1 기본 설정
├── quadrupedal_bot_trot_cfg.py     # Stage 2 Trot 설정
└── agents/
    ├── rsl_rl_ppo_cfg_stage2.py    # PPO 하이퍼파라미터
    └── rsl_rl_ppo_cfg.py           # Stage 1 PPO 설정
```

---

## 보상 구조 (Stage 2 주요 항목)

| 보상 항목 | 역할 |
|-----------|------|
| `rew_scale_lin_vel = 6.0` | 전진 속도 추적 (exp 형태) |
| `rew_scale_heading = 8.0` | 직진 유도 (sigma=0.05) |
| `rew_scale_gait = 2.0` | 대각선 trot 패턴 강제 |
| `rew_scale_air_time = 8.0` | 발 에어타임 보상 (threshold=0.12s) |
| `rew_scale_foot_height = 6.0` | 발 높이 보상 (cap=10cm) |
| `rew_scale_foot_slip = -1.5` | 슬라이딩 억제 |
| `rew_scale_diagonal_symmetry = -0.10` | FL-RR / FR-RL 관절 동기화 |
| `rew_scale_ang_vel_z = -5.0` | yaw drift 억제 |
| `rew_scale_action_rate = -0.15` | 관절 진동 억제 |

---

## 학습 이력

| 버전 | 기점 | 핵심 변경 | vel | heading_err |
|------|------|-----------|-----|-------------|
| 09-28-38 | Stage 1 | 첫 전진 보행 | ~0.3 m/s | ~10°+ |
| v12 | 09-28-38 | lin_vel 8.0, gait 1.5 → 제자리 최적해 탈출 | 0.42 m/s | 1.35° |
| v13 | 09-28-38 | air_time_threshold 0.10s, foot_slip -0.5 | 0.43 m/s | 3.63° |
| v14 | v13 | foot_slip -1.5, action_rate -0.15, entropy 복원 | 0.43 m/s | 2.20° |
| v15 | v14 | diagonal_symmetry 활성화, gait 2.0 | 0.41 m/s | 3.05° |
| v16 | v15 | heading_sigma 0.05, heading 8.0, ang_vel_z -5.0 | 진행 중 | **~1.3°** |

---

## 해결한 주요 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| 제자리 trot | gait 보상 지배 (21pt) | lin_vel=8.0, gait=1.5 |
| 셔플링 | air_time threshold 너무 작음 | threshold 0.04→0.12s |
| 대각선 미동기 | diagonal_symmetry=0.0 | -0.10 활성화 |
| heading drift | heading_sigma=0.25 → gradient 거의 0 | sigma=0.05로 gradient 7배 강화 |

---

## 현재 최신 체크포인트

```
logs/rsl_rl/spot_micro_trot/2026-05-14_13-39-47/
```

---

## 주요 저장 모델 (Best Checkpoints)

### v24b — 현재 최고 모델 ⭐
```
logs/rsl_rl/spot_micro_trot/2026-05-15_12-11-32/model_4999.pt
```

**시각화 명령어:**
```bash
DISPLAY=:1 conda run -n env_isaaclab python scripts/rsl_rl/play.py \
  --task Template-Quadrupedal-Bot-Trot-v0 \
  --num_envs 4 \
  --checkpoint /home/wodnr/quadrupedal_bot/quadrupedal_bot/logs/rsl_rl/spot_micro_trot/2026-05-15_12-11-32/model_4999.pt
```

**성능 지표 (step 4999):**
| 지표 | 값 |
|------|----|
| `actual_lin_vel_x` | 0.38 m/s |
| `heading_err_deg` | 0.76° |
| `lateral_drift_m` | 0.065 m |
| `rew/gait` | 17.4 / 20 |
| `rew/diagonal_symmetry` | -0.37 (초기 -2.28에서 개선) |
| `term_ratio` | 0.0% |

**특징:**
- 진동 감소 ✅
- 발 들기 동작 확인 ✅ (air_time 개선)
- 완벽한 직선 보행 ✅
- 걸음걸이 소폭 개선 필요 (개선 중)

**v24b 핵심 설정:**
- `gait: 5.0` (trot 페이즈 핵심 보상)
- `air_time: 8.0` + `threshold: 0.12s`
- `diagonal_symmetry: -1.0` (FL-RR/FR-RL 대각선 강제)
- `lin_vel: 6.0` + `movement: 2.0` (제자리 trot 방지)
- `heading: 12.0` + `heading_sigma: 0.025` (직선 유지)
- `pos_drift: -8.0` (누적 drift 패널티)
- `cmd_y = 0`, `cmd_ang_z = 0` (직진 전용 커리큘럼)

**전이 기저:** `2026-05-14_18-56-54/model_4999.pt`
