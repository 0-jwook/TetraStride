# Session Handoff — 2026-05-07 (최신)

## 프로젝트 개요
- **목표**: Spot Micro 12-DOF 취미 사족보행 로봇 Isaac Lab + RSL-RL PPO 학습 → 실제 로봇 배포
- **Isaac Lab 환경**: DirectRLEnv, 4096 parallel envs, RSL-RL PPO
- **URDF**: `/home/wodnr/Downloads/spot_micro_light.urdf` (2.5 kg 경량 버전)
- **링크 길이**: L1(대퇴골)=0.1075m, L2(종아리)=0.130m

---

## 현재 상태 (2026-05-07 기준)

### Stage 1 (Stance) ✅ 완료
- **ep_len=999** @ leg=0.83 rad, body_height=0.1903m, foot_spread≈0
- 체크포인트: `logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt`

### Stage 2 (Trot CCP) ⚠ 진행 중 — 품질 개선 학습 중
- **처음부터 학습** (resume=False) → Rudin 2022 방식으로 돌파구 발견
- step1에서 vel≈0.25m/s, step1000에서 vel=0.44m/s, ep_len=999, term=0 달성
- **시각화 확인 (2026-05-07)**: 덜덜 떨리며 앞으로만 나아감, 방향 휨, 자세 낮음
- 핵심 run: `logs/rsl_rl/spot_micro_trot/2026-05-07_19-05-49/model_1200.pt`
- **품질 개선 config 적용**: model_1200.pt 기반 resume 학습 예정

---

## 이번 세션에서 해결한 문제들

### 1. Stage2 서기 로컬옵티멈 탈출 (핵심 돌파구)
**문제**: Stage1 사전학습 체크포인트에서 resume 시 서기가 항상 유리
```
서기: 8.0/step × 400ep = 3200 > 이동: 16.7/step × 150ep = 2505
```
- 어떤 velocity 보상 강화도, 패널티도 서기 우위를 깰 수 없었음
- credit assignment: 이동 보상이 에피소드 초반에만 발생하면 역전파가 약함

**해결**: `resume=False` → random initialization에서 처음부터 학습
- 4096 env 중 일부가 우연히 전진 → gradient가 전체로 전파
- step1에서 이미 vel=0.25m/s 달성 (사전학습 정책 없이 더 빠른 수렴)

### 2. gait reward 항상 0 버그 수정
**문제**: 기존 코드 `rew_gait = torch.zeros(...) * scale` → 항상 0
**해결**: 올바른 gait clock 계산 로직으로 대체
```python
cos_phase = torch.cos(self._gait_phase)
pair_a = torch.tensor([1.0, 0.0, 0.0, 1.0], ...)  # FL, RR
pair_b = torch.tensor([0.0, 1.0, 1.0, 0.0], ...)  # FR, RL
contact_target = ...  # 위상에 따라 어느 발이 닿아야 하는지
contact_error = torch.abs(contact_actual - contact_target).sum(dim=1)
cmd_has_vel_gate = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
rew_gait = (4.0 - contact_error) * self.cfg.rew_scale_gait * cmd_has_vel_gate
```

### 3. lin_vel_penalty 항 추가
**추가된 것**: 속도 추적 오차 직접 패널티
```python
lin_vel_error_sq = torch.sum(
    torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1
)
rew_lin_vel_penalty = lin_vel_error_sq * self.cfg.rew_scale_lin_vel_penalty
```
현재 `rew_scale_lin_vel_penalty=0.0` (비활성, 차후 활성화 가능)

### 4. yaw 패널티 수정
**수정**: 절대값 기반 → 추적 오차 기반
```python
yaw_error = torch.square(self.robot.data.root_ang_vel_b[:, 2] - self._commands[:, 2])
rew_ang_vel_z = yaw_error * self.cfg.rew_scale_ang_vel_z
```

### 5. CUDA OOM 해결
**원인**: 이전 훈련 프로세스가 살아있는 채로 새 훈련 시작
**해결**: `kill -9`로 강제 종료 + `nvidia-smi`로 GPU 메모리 해제 확인 후 재시작

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

### quadrupedal_bot_stance_cfg.py (Stage 1 — 변경 없음)
```python
termination_height = 0.16
target_body_height = 0.19
rew_scale_body_height = 2.0
rew_scale_gravity = -2.0
rew_scale_foot_spread = -8.0
rew_scale_upright = 2.0
rew_scale_stand_still = -0.3
rew_scale_joint_default = -0.5
rew_scale_dof_pos_limits = -1.0
rew_scale_contact_forces = -1e-3
action_scale = 0.25
freeze_gait_phase = True
```

### quadrupedal_bot_trot_cfg.py (Stage 2 — 현재 최종)
```python
episode_length_s = 20.0
action_scale = 0.35
cmd_lin_vel_x_range = (0.3, 0.7)
cmd_lin_vel_y_range = (-0.2, 0.2)
cmd_ang_vel_z_range = (-0.3, 0.3)
zero_command_prob = 0.1          # 10% 서기 명령

rew_scale_alive = 0.5
rew_scale_lin_vel = 3.0
rew_scale_ang_vel = 0.3
rew_scale_ang_vel_z = -0.5       # 추적 오차 기반 yaw 패널티
rew_scale_movement = 0.0
rew_scale_lin_vel_penalty = 0.0  # 비활성

rew_scale_gait = 2.5
rew_scale_air_time = 5.0
air_time_threshold = 0.05
rew_scale_air_time_var = 1.0

rew_scale_body_height = 0.0      # ⚠ OFF — 높이 미제어 (미결 문제)
rew_scale_upright = 0.0
rew_scale_gravity = -1.0
rew_scale_ang_vel_xy = -0.05
rew_scale_lin_vel_z = -2.0
rew_scale_joint_vel = -1e-4
rew_scale_torque = -1e-5
rew_scale_action_rate = -0.01
rew_scale_termination = -5.0
rew_scale_joint_default = 0.0
rew_scale_foot_spread = 0.0
rew_scale_foot_slip = -0.05
```

### agents/rsl_rl_ppo_cfg_stage2.py (현재)
```python
num_steps_per_env = 24
max_iterations = 8000
save_interval = 200
experiment_name = "spot_micro_trot"

resume = True
load_run = "2026-05-07_19-05-49"   # 처음부터 학습: step1200에서 vel=0.44, ep=999 달성
load_checkpoint = "model_1200.pt"

policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
algorithm = RslRlPpoAlgorithmCfg(
    entropy_coef=0.01,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99, lam=0.95, desired_kl=0.01,
)
```

---

## 완료된 체크포인트

| Stage | 경로 | 주요 지표 | 비고 |
|-------|------|-----------|------|
| Stage 1 최종 | `logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt` | ep=999, h=0.1903m | kp=20, leg=0.83 ✅ |
| Stage 2 **핵심** | `logs/rsl_rl/spot_micro_trot/2026-05-07_19-05-49/model_1200.pt` | vel=0.44m/s, ep=999, term=0 | Rudin 2022 방식 ✅ |
| Stage 2 gait개선 | `logs/rsl_rl/spot_micro_trot/2026-05-07_19-50-48/model_1400.pt` | step1474 종료 | gait clock 재활성화 |
| Stage 1 구버전 | `logs/rsl_rl/spot_micro_stance/2026-04-30_02-18-41/model_4999.pt` | ep=999 | leg=0.3, kp=5 (비추천) |

---

## 미결 문제

### 1. 자세 확인 필요 (최우선)
- `body_height_mean=0.160m` (목표 0.18m 미달)
- `air_time=0.007~0.020` (발이 거의 안 들림)
- **시각화 미완료**: 서서 걷는지, 기어가는지 모름
- **play.py 올바른 실행법**:
  ```bash
  conda activate env_isaaclab
  cd /home/wodnr/quadrupedal_bot/quadrupedal_bot
  # rsl_rl_ppo_cfg_stage2.py에서 load_run/load_checkpoint 읽으므로 --checkpoint 인수 불필요
  python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4
  ```

### 2. 높이 제어 재활성화 (자세 확인 후)
만약 기어가는 것이라면:
```python
# quadrupedal_bot_trot_cfg.py
rew_scale_body_height = 1.5     # Gaussian sigma=0.05, 양수 보상
termination_height = 0.15       # 낮은 자세 종료 (현재 base_cfg 기본값 확인 필요)
```
→ 처음부터 재학습 (resume=False) 필요

### 3. Gait 품질 개선
- air_time 증가 목표: 0.020s → 0.05s 이상 (한 step = 0.02s, 최소 2~3 step 들기)
- rew_scale_air_time_var 효과 확인 (비대칭 보행 차단)

---

## 다음 단계 결정 트리

```
시각화 확인
├── 서서 걷는다 (h≥0.17m)
│   └── gait 개선 학습 계속 (현재 config에서 resume)
│       → air_time_threshold 조정, rew_scale_gait 증가
└── 기어간다 (h<0.15m)
    └── body_height 보상 재활성화
        → rew_scale_body_height=1.5, resume=False, 처음부터
```

---

## 학습 실행 명령

```bash
# conda 환경 활성화 (필수!)
conda activate env_isaaclab
cd /home/wodnr/quadrupedal_bot/quadrupedal_bot

# Stage 2 계속 (현재 config)
python scripts/rsl_rl/train.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4096 --headless

# 시각화 (play.py — --checkpoint 인수 없이 config에서 읽음)
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Trot-v0 --num_envs 4

# Stage 1 시각화
python scripts/rsl_rl/play.py --task Template-Quadrupedal-Bot-Stance-v0 --num_envs 4 \
  --checkpoint logs/rsl_rl/spot_micro_stance/2026-05-06_07-02-56/model_4999.pt
```

---

## 이번 세션 시도 이력 (누적)

| 세션 | 설정 | ep_len | 결과 |
|------|------|--------|------|
| ~44 | Stage1: kp=20, leg=0.83, shoulder effort=10 | 999 | ✅ Stage1 최종 |
| 45 | Stage2 CCP (-1.0,1.0) resume from Stage1 | ~33 | ✗ plateau (서기 로컬옵티멈) |
| 46 | Stage2 cmd범위 축소 (0.3,0.7), resume | ~33 | ✗ 여전히 plateau |
| 47 | Stage2 lin_vel=5→8, termination=-30→-50 | ~33 | ✗ 서기 우위 수식 때문에 불가 |
| 48 | **resume=False** (Rudin 2022 방식) | **999** | ✅ vel=0.44m/s 달성 |
| 49 | gait개선 (model_1200.pt에서 resume) | — | ⚠ step1474 종료 (시각화 미확인) |

---

## 환경 주의사항

- **conda 환경**: `env_isaaclab` (base 아님!)
- **URDF**: `spot_micro_light.urdf` (2.5kg, 경량 버전)
- **태스크 이름**: `Template-Quadrupedal-Bot-Stance-v0` / `Template-Quadrupedal-Bot-Trot-v0`
- **작업 디렉토리**: `/home/wodnr/quadrupedal_bot/quadrupedal_bot`
- **play.py**: `--checkpoint` 인수로 파일명만 넘기면 FileNotFoundError 발생 — config의 load_run/load_checkpoint 경로를 사용하거나 절대경로로 넘길 것
- **CUDA OOM 방지**: 새 훈련 전 `nvidia-smi`로 이전 프로세스 확인 후 `kill -9` 처리
