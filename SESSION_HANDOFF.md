# Session Handoff — 2026-05-06 (최신)

## 프로젝트 개요
- **목표**: Spot Micro 12-DOF 취미 사족보행 로봇 Isaac Lab + RSL-RL PPO 학습 → 실제 로봇 배포
- **Isaac Lab 환경**: DirectRLEnv, 4096 parallel envs, RSL-RL PPO
- **URDF**: `/home/wodnr/Downloads/spot_micro_light.urdf` (2.5 kg 경량 버전)
- **링크 길이**: L1(대퇴골)=0.1075m, L2(종아리)=0.130m

---

## 현재 상태 (2026-05-06 기준)

### Stage 1 (Stance) ✅ 완료
- **ep_len=999** @ leg=0.83 rad, body_height=0.1903m, foot_spread≈0
- 체크포인트: `logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt`

### Stage 2 (Trot CCP) ⚠ 진행 중 / 정체
- Stage 1 체크포인트에서 resume, CCP 명령 범위 (-1.0~1.0)
- ep_len ≈ 32~34 (600+ iterations 동안 정체)
- 체크포인트 디렉토리: `logs/rsl_rl/spot_micro_trot/2026-05-06_12-08-43/`
- 최신 모델: `model_5200.pt` (학습 중)

---

## 이번 세션에서 해결한 문제들

### 1. Crouch-to-Survive (크라우치 생존 전략)
**문제**: kp=5에서 로봇이 웅크려 생존 (body_height=0.155m, 목표=0.19m)
- 낮은 자세 = 중력 토크 감소 → 쉬운 균형
- gravity_sag = τ_gravity / kp = 0.206 rad (kp=5 시)

**해결**:
- kp=5 → **kp=20** (gravity_sag 0.206 → 0.052 rad)
- effort_limit=2.0 → **effort_limit=10.0** (토크 포화 63% → 17%)
- termination_height=0.12 → **0.16** (낮은 자세 생존 차단)
- target_body_height=0.18 → **0.19** (실로봇 실측 목표)

### 2. 앞다리 중앙 모임 (어깨 약함)
**문제**: 왼쪽 앞다리가 몸통 중앙으로 이동해 중심 잡기
- shoulder kp=15, effort=2.0 N·m: 물리적으로 너무 약함
- 다리를 벌리면 페널티 → 정책이 강제로 모음

**해결**:
- shoulder effort_limit=2.0 → **10.0**
- shoulder kp=15 → **20** (leg/foot과 통일)
- rew_scale_foot_spread: -2 → **-8** (Stage1), **-5** (Stage2)

### 3. 실제 로봇 자세 (leg=0.3→leg=0.83)
**문제**: leg=0.3, foot=-0.3은 너무 펴진 자세 (body_height=0.233m)
- 실 로봇 측정: Q2=-0.83 rad, Q3=1.66 rad → 시뮬 leg=0.83, foot=-0.83

**해결**: leg=0.3, foot=-0.3, kp=5 → **leg=0.83, foot=-0.83, kp=20, effort=10**

---

## 핵심 물리 공식

```
gravity_sag = τ_gravity / kp                    # 정적 관절 처짐
τ_gravity   = m×g×L1×sin(leg) / 4legs           # hip 중력 토크
body_height = L1×cos(leg) + L2×cos(leg+foot)    # 몸통 높이

# kp=20, leg=0.83 기준:
#   τ_gravity ≈ 2.5×9.81×0.1075×sin(0.83)/4 ≈ 0.38 N·m
#   gravity_sag ≈ 0.38/20 ≈ 0.019 rad (negligible)
#   body_height ≈ 0.1075×cos(0.83) + 0.130×cos(0.83-0.83) ≈ 0.197m
```

---

## 현재 파일 상태

### spot_micro_cfg.py
```python
pos=(0.0, 0.0, 0.22)
joint_pos={".*_shoulder": 0.0, ".*_leg": 0.83, ".*_foot": -0.83}
# 모든 관절 통일:
effort_limit=10.0, saturation_effort=10.0, velocity_limit=6.0
stiffness=20.0, damping=0.5 (foot: damping=0.6)
soft_joint_pos_limit_factor=0.9
```

### quadrupedal_bot_stance_cfg.py (Stage 1)
```python
termination_height = 0.16       # 낮은 자세 생존 차단
target_body_height = 0.19       # 실로봇 실측 목표
rew_scale_body_height = 2.0     # Gaussian sigma=0.05
rew_scale_gravity = -2.0
rew_scale_foot_spread = -8.0    # 다리 모임 강하게 차단
rew_scale_upright = 2.0
rew_scale_stand_still = -0.3
rew_scale_joint_default = -0.5  # 어깨 이탈 패널티
rew_scale_dof_pos_limits = -1.0
rew_scale_contact_forces = -1e-3
action_scale = 0.25
freeze_gait_phase = True
```

### quadrupedal_bot_trot_cfg.py (Stage 2)
```python
cmd_lin_vel_x_range = (-1.0, 1.0)   # CCP 전방향
cmd_lin_vel_y_range = (-0.5, 0.5)
cmd_ang_vel_z_range = (-1.0, 1.0)
zero_command_prob = 0.02             # 2% 제자리 서기 병행
rew_scale_lin_vel = 4.0
rew_scale_foot_spread = -5.0
rew_scale_movement = 3.0
rew_scale_gait = 3.0
```

### agents/rsl_rl_ppo_cfg_stage1.py
```python
max_iterations = 5000
entropy_coef = 0.01         # 0.005→0.05(너무큼)→0.01
```

### agents/rsl_rl_ppo_cfg_stage2.py
```python
resume = True
load_run = "2026-05-06_07-02-56"    # 어깨 수정 + foot_spread=-8 stance 최종
load_checkpoint = "model_4999.pt"
max_iterations = 6000
entropy_coef = 0.002
```

---

## 완료된 체크포인트

| Stage | 경로 | ep_len | 비고 |
|-------|------|--------|------|
| Stage 1 **최신 최종** | `logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt` | 999 | kp=20, leg=0.83 ✅ |
| Stage 2 (진행 중) | `logs/rsl_rl/spot_micro_trot/2026-05-06_12-08-43/model_5200.pt` | ~33 | CCP plateau |
| Stage 1 (구버전) | `logs/rsl_rl/spot_micro_stance/2026-04-30_02-18-41/model_4999.pt` | 999 | leg=0.3, kp=5 (비추천) |

---

## Stage 2 정체 원인 및 다음 단계

### 현재 진단 (ep_len≈32 정체)
- CCP 명령 범위 (-1.0, 1.0)이 Stance 정책에서 직접 trot로 전환하기엔 너무 넓음
- rew/body_height ≈ -5.0 → Stage2 trot_cfg에 body_height 목표 미설정 (base_cfg 기본값 -8.0 적용)
- gait reward가 trot 패턴을 잘 포착하지 못할 수 있음

### 권장 다음 단계

**옵션 A — CCP 범위 좁히기 (보수적 접근)**
```python
# quadrupedal_bot_trot_cfg.py 수정
cmd_lin_vel_x_range = (0.0, 0.5)   # (-1.0,1.0) → 전진만
cmd_lin_vel_y_range = (0.0, 0.0)
cmd_ang_vel_z_range = (0.0, 0.0)
```
→ 간단한 전진 학습 먼저, 이후 점차 확대

**옵션 B — Stage 2 보상 재조정**
```python
rew_scale_body_height = 1.5     # Stance처럼 Gaussian 양수로 전환
rew_scale_lin_vel = 5.0         # 속도 추적 더 강화
rew_scale_termination = -10.0   # 종료 패널티 추가
```

**옵션 C — 현재 Stage 2 완료 후 다음 세션에서 위 수정 적용**

---

## 학습 실행 명령

```bash
# conda 환경 활성화 (필수!)
conda activate env_isaaclab

# Stage 1 서기 학습
python scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4096 --headless

# Stage 2 트롯 학습
python scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4096 --headless

# 시각화
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt

python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_trot/2026-05-06_12-08-43/model_5200.pt
```

---

## 이번 세션 시도 이력 (누적)

| 세션 | 설정 | ep_len | 결과 |
|------|------|--------|------|
| 32 | leg=0.3, foot=-0.3, kp=5, z=0.25 | 999 | ✅ 구버전 성공 |
| 40 | kp=20, effort=10, leg=0.83 | ~200 | ⚠ torque_sat=17% |
| 41 | entropy=0.05 (실수) | ~30 | ✗ 수렴 실패 |
| 42 | entropy=0.01 수정, kp=20 | 999 | ✅ body_height=0.155m (낮음) |
| 43 | termination=0.16, body_height_target=0.19 | 999 | ✅ 0.1903m ✅ |
| 44 | shoulder effort=10, kp=20, foot_spread=-8 | 999 | ✅ Stage1 최종 |
| 45 | Stage2 CCP (-1.0,1.0) resume from 44 | ~33 | ⚠ plateau 진행 중 |

---

## 환경 주의사항

- **conda 환경**: `env_isaaclab` (base 아님!)
- **URDF**: `spot_micro_light.urdf` (2.5kg, 경량 버전)
- **태스크 이름**: `Template-Quadrupedal-Bot-Stance-v0` / `Template-Quadrupedal-Bot-Trot-v0`
- **작업 디렉토리**: `/home/wodnr/quadrupedal_bot/quadrupedal_bot`
