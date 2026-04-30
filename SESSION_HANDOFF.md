# Session Handoff — 2026-04-30

## 프로젝트 개요
- **목표**: Spot Micro 12-DOF 취미 사족보행 로봇 Isaac Lab + RSL-RL PPO 학습 → 실제 로봇 배포
- **Isaac Lab 환경**: DirectRLEnv, 4096 parallel envs, RSL-RL PPO
- **URDF**: `/home/wodnr/Downloads/spot_micro.urdf` | 총 질량 5.3 kg
- **링크 길이**: L1(대퇴골)=0.1075m, L2(종아리)=0.130m

---

## 현재 상태

**Stage 1 (Stance) 재완료 — ep_len=999 달성**

낮은 높이(0.18m) 시도가 모두 실패하여 유일하게 안정적인 설정(leg=0.3, foot=-0.3)으로 복구함.

---

## 핵심 물리 공식

```
z_body  = L1×cos(leg) + L2×cos(leg+foot)   # 몸 높이
x_toe   = L1×sin(leg) + L2×sin(leg+foot)   # 발 수평 위치 (0=균형)
hip 중력 토크 = m×g×L1×sin(leg) / 4legs   # 엉덩이 관절 부하
무릎 모멘트 = F_contact × L2 × sin(|calf_angle|)
```

**안정 조건**: calf_angle=0 AND hip 토크 < ~40% effort_limit
**불안정**: calf_angle≠0 OR hip 토크 과대(leg 각도 너무 클 때)

---

## 물리적 한계 — 달성 가능 높이

| 목표 높이 | leg 각도 | hip 중력 토크 | effort_limit 대비 | 결과 |
|-----------|---------|--------------|-----------------|------|
| 0.18m | 62° | 1.24 N·m | 62% | ✗ 포화 |
| 0.20m | 49° | 1.05 N·m | 52% | ✗ 포화 |
| 0.22m | 35° | 0.81 N·m | 41% | ✗ 아슬 |
| **0.233m** | **17°** | **0.41 N·m** | **21%** | **✓ 유일 성공** |

0.233m 이하 목표는 kp=5, effort_limit=2.0 N·m 조건에서 모두 실패. 더 낮은 높이가 필요하면 effort_limit 증가(≥4 N·m) 또는 kp 대폭 증가 필요.

---

## 이번 세션 시도 이력

| Session | 설정 | ep_len | 결과 |
|---------|------|--------|------|
| 29 | leg=0.4, foot=-0.73, kp=12, z=0.222 | ~56 | ✗ body_height 0.178m 고착 |
| 30 | leg=0.873, foot=-1.559, kp=12, z=0.17 | ~21 | ✗ 가장 나쁨 |
| 31 | leg=1.087, foot=-1.087, calf=0°, kp=5, z=0.18 | 32 | ✗ hip 토크 62% 포화 |
| **32** | **leg=0.3, foot=-0.3, kp=5, z=0.25** | **999** | **✓ 성공** |

---

## 현재 파일 상태 (Session 32 성공 설정)

### spot_micro_cfg.py — 현재 상태
```python
pos=(0.0, 0.0, 0.25)
joint_pos={".*_shoulder": 0.0, ".*_leg": 0.3, ".*_foot": -0.3}
leg_joints:  stiffness=5.0, damping=0.25
foot_joints: stiffness=5.0, damping=0.25
shoulder_joints: stiffness=5.0, damping=0.25
effort_limit=2.0, velocity_limit=6.0
```

### quadrupedal_bot_stance_cfg.py — 현재 상태
```python
target_body_height = 0.233
termination_height = 0.12
rew_scale_body_height = -5.0
freeze_gait_phase = True
```

---

## 완료된 체크포인트

| Stage | 경로 | 비고 |
|-------|------|------|
| Stage 1 (Session 32) | `logs/rsl_rl/spot_micro_stance/2026-04-30_02-18-41/model_4999.pt` | ep_len=999, **최신** |
| Stage 1 (Session 34 이전) | `logs/rsl_rl/spot_micro_stance/2026-04-29_12-42-17/model_4999.pt` | ep_len=999, 이전 |
| Stage 2 (Trot) | `logs/rsl_rl/spot_micro_trot/2026-04-28_23-52-35/model_10998.pt` | ep_len=495~499, 이전 Stage1 기반 |

---

## 다음 단계 옵션

### A. Stage 2 재학습 (Stage 1 Session 32 체크포인트 기반)
```bash
python scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4096 --headless
```
trot_cfg.py에서 Stage 1 체크포인트 경로 업데이트 필요.

### B. 기존 Stage 2 체크포인트 재사용
로봇 설정이 동일(leg=0.3, foot=-0.3, kp=5)하므로 기존 trot 체크포인트 그대로 사용 가능.

### C. 실제 로봇 배포 시작
```bash
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_stance/2026-04-30_02-18-41/model_4999.pt
```

---

## 시각화 명령
```bash
# Stage 1 (최신)
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_stance/2026-04-30_02-18-41/model_4999.pt

# Stage 2
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_trot/2026-04-28_23-52-35/model_10998.pt
```
