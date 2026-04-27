# Spot Micro 강화학습 학습 히스토리 & 오류 분석

> 작성일: 2026-04-27  
> 프로젝트: 4족 보행 로봇 (Spot Micro, 12 DOF) — Isaac Lab + RSL-RL PPO  
> 목표: 커리큘럼 학습 Stage1(서기) → Stage2(트롯) → Stage3(전속도 보행)

---

## 1. 환경 기본 정보

| 항목 | 값 |
|------|-----|
| 로봇 | Spot Micro (hobby quadruped, 12 DOF) |
| 시뮬레이터 | Isaac Lab (PhysX 기반) |
| RL 프레임워크 | RSL-RL (PPO) |
| 물리 dt | 1/200 s |
| decimation | 4 → control freq 50 Hz |
| step_dt | 0.02 s |
| 관절 구조 | shoulder(abduction) + leg(hip flex) + foot(knee) × 4 |
| 초기 높이 | 0.25 m (서있을 때 base_link ≈ 0.22 m) |
| 병렬 환경 수 | 4096 |

### 관절 제한 범위
```
shoulder: ±0.548 rad
leg:      [-2.666, 1.548] rad
foot:     [-2.600, 0.100] rad
```

### Standing pose 기준
```
leg = 0.5 rad, foot = -0.5 rad
Vertical reach: cos(0.5)*0.1075 + cos(0)*0.130 ≈ 0.224 m
```

---

## 2. 전체 학습 타임라인

```
Phase A: 초기 탐색 (1~9차)
  │  └ 보상 설계 미숙, standing local optimum 반복
  │
Phase B: legged_gym 기반 재설계 (10~15차)
  │  └ velocity bootstrap, cmd tracking 도입
  │  └ 15차: Stage1 ep_len 212 달성 (최초 성공)
  │
Phase C: 외부 리뷰 기반 구조 수정 시도 (16~18차)
     └ 구조 버그 수정 → 새 버그 도입 → ep_len 역행 (현재)
```

---

## 3. Phase A: 초기 탐색 (1~9차, commit 34ce4d4 ~ 9adb712)

### 3.1 초기 설계 (1~5차)

#### 문제: Standing local optimum
로봇이 4발로 바닥에 누워 미끄러지는 방식으로 속도 보상을 취함.
- 이동 없이 관절을 작동해 speed reward를 0으로 유지하는 게 최적
- Contact sensor 없이 속도만 추적 → 서기/걷기 구분 불가

#### 시도했던 수정들
- ContactSensor 추가/제거 반복
- feet air time reward 추가 (threshold 0.5s → 0.2s 조정)
- 리셋 노이즈 ±0.05 → ±0.2 rad
- 속도 보상 형태 변경 (Gaussian exp → linear)

#### 결과
- ep_len 200~300 수준에서 정체
- 실제 보행이 아닌 sliding/stationary policy로 수렴

---

### 3.2 legged_gym 방식 도입 (10~11차, commit 6a5713b)

**분기점: 10차** — legged_gym (Rudin 2021) 기반으로 전면 재설계

```python
# 10차 주요 변경
rew_alive = 0         # 생존 보상 제거
rew_termination = 0   # 종료 패널티 없음
exp tracking reward   # Gaussian exp(-error/sigma) 형태
velocity bootstrap    # 리셋 시 로봇에 명령 속도 부여
only_positive_rewards # compute_rewards에 clamp(min=0.0) 적용
```

**velocity bootstrap**: 리셋 시 `root_state[:, 7] = cmd_vel_x`로 초기 속도 부여
→ 보상 신호 강화 효과, 하지만 "공짜 속도"로 실제 학습 없이 보상 수령 가능

**결과**: ep_len 향상, 하지만 구조적 문제 잠재

---

### 3.3 gait phase 도입 (12~15차, commit 55f93eb ~ 55c1aa7)

**분기점: 12차** — observation에 gait_phase(sin/cos) 추가, obs 50차원

```python
observation_space = 50  # +2 (gait phase sin/cos)
```

**분기점: 커리큘럼 구조 도입** (commit 24bfa0d)
```
Stage1: Template-Quadrupedal-Bot-Stance-v0  (서기, cmd=0)
Stage2: Template-Quadrupedal-Bot-Trot-v0   (트롯, cmd 0.1~0.4 m/s)
Stage3: Template-Quadrupedal-Bot-Direct-v0  (전속도, cmd 0.2~0.6 m/s)
```

**반복되는 문제: entropy collapse**
```
entropy_coef 조정 이력:
0.005 → 0.001 → 0.01 → 0.003 → 0.002 → 0.005
```
entropy가 음수로 떨어지면 policy가 deterministic하게 고정 → 탐색 불가 → plateau

**분기점: 15차 (commit 55c1aa7)** — 최초 Stage1 성공

```python
# 15차 핵심 설정
stiffness = 60.0, damping = 3.0     # PD gain
termination_height = 0.15           # 높이 0.15m 미만 종료
body_tilted = gravity_b[:, 2] > -0.5  # 60도 이상 기울면 종료
rew_scale_termination = 0.0         # 종료 패널티 없음
compute_rewards return total.clamp(min=0.0)  # 음수 보상 방지

# 새로 추가된 보상
rew_scale_upright = 1.0             # exp(-tilt/0.04) 형태
rew_scale_joint_default = -0.5      # 어깨 ±0.2 rad 초과 시 패널티
rew_scale_lin_vel_xy = -0.3         # 수평 이동 패널티 (stance)
rew_scale_ang_vel_z = -0.3          # yaw 스핀 패널티
```

**결과**: Stage1 ep_len **212** 달성 (5000 iter 기준). Stage2, Stage3는 실패.

---

## 4. Phase B: Stage2/3 실패 분석 (commit baf4a01 이전 상태)

### Stage2 실패 원인 (당시 분석)
1. `trot_cfg.py`에 `rew_scale_upright`, `rew_scale_ang_vel_z`, `rew_scale_joint_default` 누락 → base cfg의 0.0 상속
2. gait scale이 5.0으로 너무 강해 contact schedule 강제 → 불안정 유발

### Stage3 실패 원인
1. 실패한 Stage2 체크포인트 로드
2. 속도 명령 (0.5~1.0 m/s)이 못 걷는 로봇에게 너무 강함
3. `rew_scale_upright`, `rew_scale_alive` = 0 → 자세 유지 인센티브 없음

---

## 5. Phase C: 외부 리뷰 기반 구조 수정 (16~18차)

### 5.1 외부 리뷰 1: Rudin 2021 표준 위반 지적

**지적 사항**:
1. `compute_rewards`의 `clamp(min=0.0)` → 패널티 gradient 전부 차단
2. `rew_scale_termination = 0.0` → 조기 종료 패널티 없음 → Rudin 표준 -200
3. velocity bootstrap (`root_state[:, 7] = cmd_vel`) → 공짜 속도 보상
4. termination 조건 너무 strict: height 0.15m (Rudin 표준 0.05m)

**분기점: commit baf4a01** (구조적 결함 수정)
```python
# 변경 전 → 후
return total.clamp(min=0.0)  →  return total        # clamp 제거
rew_scale_termination = 0.0  →  -200.0              # Rudin 표준
termination_height = 0.15    →  0.05                # Rudin 표준
body_tilted = gravity_b > -0.5  →  > 0.0            # 더 관대하게
velocity bootstrap 제거                              # 정지 리셋
action_scale = 0.5           →  0.25                # Rudin 표준
init_noise_std = 1.0         →  0.6                 # 카오스 방지
dof_acc 패널티 추가: -2.5e-7                         # 관절 진동 억제
```

---

### 5.2 외부 리뷰 2: Sliding policy + Shivering 분석

**Sliding policy 분석**: 4발이 모두 지면에 있는 상태에서 gait reward = 0 (패널티 없음)
→ 로봇이 바닥에서 미끄러지면서 속도 보상을 얻는 것이 optimal

**Shivering policy 분석**: action_scale 0.5 + 50Hz control = 관절이 28°/step 진동 가능

**분기점: commit b08e5c6** (Margolis 2022 표준 추가)
```python
# 신규 보상 추가
rew_foot_slip: -0.05    # 접지 발 수평 속도 패널티 (sliding 방지 핵심)
rew_no_air: -1.0        # 4발 전부 접지 패널티 (air phase 강제)
rew_foot_clearance: -2.0  # swing foot 목표 높이 6cm (Margolis 2022)
rew_stand_still: -0.5   # cmd=0 시 관절 default 이탈 패널티
rew_action_acc: -0.005  # action 2차 미분 패널티 (jitter 억제)
rew_foot_spread         # X축 추가 (Y축만 → X+Y 양방향)

# EMA action smoothing (Margolis 2022)
processed = 0.8 * action + 0.2 * processed_prev
target = default_joint_pos + processed * 0.25

# PD gain 완화 (hobby robot 적정값)
stiffness 60 → 30, damping 3 → 1.5
```

---

### 5.3 16차 학습 결과 (commit b08e5c6 이후)

| iter | ep_len | mean_reward | entropy | action_std |
|------|--------|-------------|---------|------------|
| 1 | 72.6 | -1970 | 11.08 | 0.61 |
| 5 | 81.3 | -2233 | 11.15 | 0.61 |
| 205 | **15.6** | -328 | 13.87 | 0.79 |

**결과**: ep_len이 72 → **15로 역행** (early death trap 발생)

**원인 분석**:
```
per-step reward ≈ -20 (음수 패널티 합 과다)
termination -200 (한 번)

ep_len 16 살면: -20 × 16 = -320 + (-200) = -520
ep_len 100 살면: -20 × 100 = -2000 + (-200) = -2200

→ PPO 입장에서 빨리 죽는 것이 유리 → ep_len 역행
```

---

### 5.4 17차 학습 결과 (commit 220baa2, per-step 패널티 완화)

```python
# stance_cfg 변경
rew_scale_alive: 0.5 → 1.0
rew_scale_gravity: -5.0 → -2.0
rew_scale_foot_spread: -2.0 → -0.5
rew_scale_stand_still: -0.5 → -0.1
rew_scale_lin_vel_xy: -0.3 → -0.1
rew_scale_ang_vel_z: -0.3 → -0.1
```

| iter | ep_len | mean_reward | entropy | action_std |
|------|--------|-------------|---------|------------|
| 6 | 75.3 | -714 | 11.20 | 0.62 |
| 152 | 85.2 | -631 | 15.09 | 0.85 |
| 325 | **46.0** | -197 | 12.25 | 0.72 |

**결과**: early death trap 재발. ep_len 75 → **46으로 역행**

---

### 5.5 18차 학습 (commit 4bc87f7, PD gain 복원 + termination 수정)

**추가 진단으로 발견한 근본 원인**:

```python
# 15차 (성공)와 현재의 핵심 차이
15차: stiffness=60, damping=3    ← 초기 자세 복원력 충분
현재: stiffness=30, damping=1.5  ← 초기 random policy에서 자세 유지 불가

# projected_gravity_b는 normalized 벡터 (크기 1, Isaac Lab 확인)
# > 0.0 조건은 90도 이상 기울 때만 종료 → 너무 관대
# > -0.5 조건은 60도 이상 기울 때 종료 → 15차 기준
```

**분기점: commit 4bc87f7** — 근본 원인 수정
```python
# 수정 1: PD gain 복원
stiffness 30 → 60, damping 1.5 → 3.0

# 수정 2: tilting 종료 조건 복원
body_tilted = gravity_b[:, 2] > -0.5  # 60도 기준 복원

# 수정 3: compute_rewards 재구성
# 기존: return total (clamp 없음, termination도 총합에 포함)
# 문제: termination -200이 per-step 패널티에 묻힘
# 수정: living.clamp(min=0.0) + rew_termination
#       → 생존 보상은 0 이상 보장, 종료 패널티는 항상 작동
```

**18차 결과** (진행 중):

| iter | ep_len | mean_reward | entropy | action_std |
|------|--------|-------------|---------|------------|
| 6 | 54.0 | -452 | 11.19 | 0.62 |
| 125 | 37.3 | -271 | 15.34 | 0.87 |

**아직 ep_len 역행 중** → 추가 진단 필요

---

## 6. 발견된 버그 목록 (우선순위 순)

### 🔴 Critical (학습 실패 직접 원인)

#### Bug 1: Early Death Trap
**현상**: ep_len이 수백 iteration 후 15~50으로 역행  
**원인**: per-step 패널티 합 > termination penalty / max_episode_length  
**공식**: `sum(per_step_penalty) × ep_len > |termination_penalty|` 이면 조기 사망이 유리  
**수식 예시**:
```
per-step: -4.4, ep_len=45 → total = -4.4×45 = -198 ≈ termination(-200)
→ 경계에서 PPO가 어느 쪽이든 비슷 → 불안정 수렴
```
**해결 방향**: 
- `living.clamp(min=0) + rew_termination` 구조 사용 (현재 적용)
- 또는 per-step alive 보상을 `termination_penalty / max_ep_len`보다 크게 설정

#### Bug 2: compute_rewards clamp가 패널티를 전부 무력화
**현상**: gravity, ang_vel_xy 등 6개 패널티가 학습에 영향을 주지 못함  
**원인**: `total.clamp(min=0.0)` → total이 음수가 될 수 없어 패널티 gradient = 0  
**구체적 예시**:
```python
alive(0.5) + upright(0.3) + gravity(-2.0 * tilt) + ... = 예를 들어 0.8 - 1.5 = -0.7
clamp(min=0) → 0 (gradient 없음)
```
**해결**: `living.clamp(min=0) + rew_termination` 구조

#### Bug 3: Velocity Bootstrap
**현상**: 리셋 시 로봇에게 명령 속도를 초기 속도로 부여 → 속도 추적 보상이 공짜  
**코드 위치**:
```python
# _reset_idx (구 코드)
root_state[:, 7] = self._commands[env_ids, 0]  # vx
root_state[:, 8] = self._commands[env_ids, 1]  # vy
```
**결과**: 로봇이 실제로 걷는 법을 배우지 않아도 ep 초반에 높은 보상 수령  
**해결**: 제거 (현재 적용)

---

### 🟡 Major (학습 품질 저하)

#### Bug 4: Sliding Survival Policy
**현상**: 4발 모두 지면에 닿은 채 미끄러지며 속도 보상 취득  
**원인**: gait_reward가 4발 접지 시 0 (패널티 없음) → sliding이 걷는 것보다 쉬움  
**해결**:
```python
rew_foot_slip: contact 발의 수평 속도 × -0.05  (미끄러지면 패널티)
rew_no_air: 4발 전부 접지 시 -1.0  (air phase 강제)
```

#### Bug 5: PD Gain 과도한 하향 (stiffness 60→30)
**현상**: 초기 random policy에서 관절 노이즈(±0.1 rad)를 복원하지 못하고 즉시 쓰러짐  
**원인**: hobby robot에 stiffness 30은 너무 낮음 — 자세 복원 토크 부족  
**증거**: 15차(stiffness 60) ep_len 212, 17차(stiffness 30) ep_len 45  
**해결**: 60 복원 (현재 적용)

#### Bug 6: Shivering Policy (진동 액션)
**현상**: 관절이 고주파로 진동 → 실제 이동 없이 action noise로 인한 토크 소비  
**원인**: action_scale 0.5 + 50Hz = 관절당 28°/step 진동 가능  
**해결**:
```python
action_scale: 0.5 → 0.25
EMA smoothing: processed = 0.8 * action + 0.2 * processed_prev
action_acc penalty: -0.005
dof_acc penalty: -2.5e-7
```

---

### 🟢 Minor (보상 설계 개선)

#### Issue 7: upright 보상 sigma 값이 gravity 벡터 단위와 불일치 가능성
**코드**:
```python
tilt = sum(square(gravity_b[:, :2]))
rew_upright = exp(-tilt / 0.04)
```
**확인**: Isaac Lab `projected_gravity_b`는 **normalized 벡터 (크기 1)** 
→ 0.04 sigma는 적정 (약 11.5도 이상 기울면 보상 급감)
→ 버그 아님, 단 너무 날카로울 수 있음

#### Issue 8: air_time threshold 0.4s → stance에서 0.0 scale로 비활성화되어 있어 무관
**코드**: `rew_scale_air_time = 0.0` (stance_cfg) → 0.1→0.4 변경의 실효 없음

---

## 7. 설정값 변화 추적 (핵심 파라미터)

### action_scale
```
초기: 0.5 (너무 큼)
baf4a01: 0.5 → 0.25 (Rudin 표준)
현재: 0.25
```

### PD Gain (leg, foot stiffness)
```
초기: 10 (너무 낮음, 물리 버그 수정)
ac0e354: 10 → 30
55c1aa7 (15차): 30 → 60 (성공)
b08e5c6 (16차): 60 → 30 (실패 원인)
4bc87f7 (18차): 30 → 60 (복원)
```

### termination_height
```
초기: 0.15 m
baf4a01: 0.15 → 0.05 m (Rudin 표준)
현재: 0.05 m
```

### body_tilted (종료 조건)
```
초기: gravity_b[:, 2] > -0.5  (60도 이상)
baf4a01: > -0.5 → > 0.0  (90도, 너무 관대)
4bc87f7: 0.0 → -0.5  (60도 복원)
```

### rew_scale_termination
```
15차: 0.0
baf4a01: 0 → -200.0
현재: -200.0
```

### compute_rewards return
```
초기 ~ 15차: total.clamp(min=0.0)
baf4a01 ~ b08e5c6: total (clamp 없음, 패널티 작동하지만 early death trap)
4bc87f7: living.clamp(min=0.0) + rew_termination (절충)
```

### entropy_coef
```
초기: 0.005
6차: → 0.001 (std 폭주 방지)
7차: → 0.002
baf4a01: 0.002 → 0.005 (stage1)
현재: 0.005 (stage1)
```

---

## 8. 커리큘럼 Stage별 현재 설정 요약

### Stage1 (서기, stance_cfg.py)
| 보상 | 스케일 | 비고 |
|------|--------|------|
| alive | +1.0 | 생존 인센티브 |
| upright | +1.0 | exp(-tilt/0.04) |
| gravity | -2.0 | 기울기 패널티 |
| lin_vel_xy | -0.1 | 이동 방지 |
| ang_vel_z | -0.1 | yaw 방지 |
| joint_default | -0.2 | 어깨 ±0.2 rad 초과 |
| foot_spread | -0.5 | Y+X 간격 |
| foot_slip | -0.02 | 미끄러짐 |
| stand_still | -0.1 | cmd=0 시 관절 이탈 |
| termination | -200 | 조기 종료 억제 |
| gait/air_time/no_air | 0.0 | Stage2에서 활성화 |

### Stage2 (트롯, trot_cfg.py)
| 보상 | 스케일 | 비고 |
|------|--------|------|
| lin_vel | +1.5 | 속도 추적 |
| gait | +2.0 | trot contact schedule |
| air_time | +3.0 | 발 들기 보상 |
| no_air | -1.0 | 4발 전부 접지 방지 |
| foot_clearance | -2.0 | swing foot 6cm |
| foot_slip | -0.05 | 미끄러짐 |
| upright | +0.5 | 직립 유지 |

### Stage3 (전속도, env_cfg.py base)
| 설정 | 값 |
|------|-----|
| cmd_lin_vel_x | (0.2, 0.6) m/s |
| cmd_lin_vel_y | (-0.3, 0.3) m/s |
| cmd_ang_vel_z | (-0.5, 0.5) rad/s |

---

## 9. 현재 미해결 문제 (2026-04-27 기준)

### 문제 1: 18차에서도 ep_len 역행 중
- 18차 시작: ep_len 54 → 125 iter에서 37로 역행
- stiffness 60 복원, clamp 절충 적용에도 불구하고 지속
- **의심 원인**: `_get_observations`와 `_get_rewards` 호출 순서 문제?
  - `_get_observations`에서 `self._last_actions = self.actions.clone()` 업데이트
  - 만약 `_get_observations` → `_get_rewards` 순서라면 `action_acc` 계산 오류

### 문제 2: 15차와 현재 코드 차이점 미확인 항목
15차에서 ep_len 212 달성 시 없었던 것들:
- EMA action smoothing (`processed_actions`)
- foot_slip, foot_clearance, no_air, stand_still, action_acc 패널티
- `_last_last_actions`, `_last_joint_vel` 새 state tensor

이 중 하나가 새로운 버그를 도입했을 가능성 있음.

### 문제 3: 관련 확인 필요 사항
```python
# Isaac Lab DirectRLEnv step() 내 실제 호출 순서
# → _get_dones → _get_rewards → _get_observations 순인지 확인 필요
# → _last_actions가 reward 계산 시점에 올바른 값인지 확인
```

---

## 10. 앞으로의 방향 제안

### Option A: 15차 코드 베이스로 복원 후 점진적 추가 (권장)
15차에서 ep_len 212가 됐던 코드는 작동 확인된 상태이다.
새 기능을 하나씩 추가하며 어느 시점에 실패하는지 이분 탐색.

```
Step 1: 15차 코드로 되돌리기 (git checkout 55c1aa7 -- env.py)
Step 2: 학습 → ep_len 212 재현 확인
Step 3: velocity bootstrap 제거만 추가 → ep_len 확인
Step 4: action smoothing 추가 → 확인
Step 5: foot_slip 추가 → 확인
Step 6: termination penalty -200 추가 + living.clamp + rew_termination 구조 → 확인
```

### Option B: 현재 코드 디버깅 (신중 접근)
18차가 실패하는 정확한 원인 파악:

```python
# 진단 코드 추가 (임시)
# _get_rewards에서 주요 보상 항목별 값 출력
print(f"alive={rew_alive.mean():.3f}, gravity={rew_gravity.mean():.3f}, "
      f"upright={rew_upright.mean():.3f}, slip={rew_foot_slip.mean():.3f}")
```

### Option C: 패널티 완전 제거 후 점진적 추가
stance_cfg의 모든 패널티를 0으로 만들고, alive + upright만으로 학습 시작.
수렴 확인 후 패널티를 하나씩 추가.

---

## 11. 체크포인트 정보

| Stage | 위치 | 성능 | 비고 |
|-------|------|------|------|
| Stage1 (최고) | `logs/rsl_rl/spot_micro_stance/stage1_stance` (→ 2026-04-27_00-55-30) | ep_len ~212 | 15차 코드 기준 |
| Stage1 최신 | `model_4999.pt` | ep_len ~212 | 사용 가능 |

---

## 12. 핵심 교훈

### 교훈 1: 한 번에 너무 많이 바꾸지 말 것
15차→16차에서 다음을 동시에 변경:
- PD gain 변경 (60→30)
- clamp 제거
- termination penalty 추가 (-200)
- tilting 조건 변경
- 5가지 새 보상 추가
- action smoothing 추가
→ 어느 변경이 실패 원인인지 알 수 없음

### 교훈 2: 작동하는 코드는 하나씩만 바꿀 것
15차 ep_len 212가 기준점. 이것을 유지하면서 개선해야 함.

### 교훈 3: Early Death Trap 공식
```
per_step_net_reward × max_episode_length < |termination_penalty|
이 조건이 성립하면 생존이 항상 유리 → early death trap 없음
```

현재 설정에서 검증:
```
alive(1.0) + upright(~0.5 평균) = ~1.5 per step (양수)
gravity(-2.0 × tilt) + ... ≈ ?

최악의 경우 net per-step ≈ +1.5 - 3.0 = -1.5
max_ep_len = 1000 step
-1.5 × 1000 = -1500 >> 200 (termination)
→ 여전히 early death trap 가능성
```

**근본 해결**: alive 보상을 `|termination| / max_ep_len = 200/1000 = 0.2`보다 크게,
그리고 per-step net이 양수가 되도록 패널티 조정.

### 교훈 4: PD gain과 학습 난이도
- 높은 stiffness → 초기 자세 안정적 → 학습 쉬움 (but 실제 로봇에서 shivering 위험)
- 낮은 stiffness → 초기 불안정 → 학습 어려움 (but 더 자연스러운 동작)
- **학습 초기에는 높은 stiffness로 수렴 후, 실제 배포 전에 낮추는 전략 유효**

### 교훈 5: Rudin 2021 표준을 그대로 적용하면 안 되는 경우
Rudin의 termination -200은 그들의 보상 scale(alive ~1.0, tracking ~1.0)에 맞춘 값.
hobby robot의 작은 스케일에서는 다른 값이 적합할 수 있음.
→ termination penalty는 항상 `|termination| < alive × max_ep_len`이 되도록 설정.
