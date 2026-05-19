# Spot Micro RL — 물리적 적응 + AI 에이전트 협업 프로토콜

> 작성일: 2026-04-27
> 기반 문서: `TRAINING_HISTORY.md`, `FUTURE_DIRECTION.md`
> 목적: ① 논문 파라미터를 Spot Micro의 물리에 맞게 재산정하는 절차, ② AI 에이전트와 협업하는 의사결정 프로토콜

---

## 0. 왜 이 문서가 필요한가

`FUTURE_DIRECTION.md`에서 Rudin 2021의 termination -2.0이 표준이라고 정리했지만, 그것조차 **ANYmal (30 kg, torque-controlled BLDC)을 기준으로 한 값**이다. Spot Micro는 다음 점에서 본질적으로 다른 시스템이다:

| 항목 | ANYmal (Rudin) | Mini Cheetah (Margolis) | **Spot Micro** |
|------|---------------|------------------------|----------------|
| 총 질량 | ~30 kg | ~9 kg | **~1.5 kg** |
| 다리 길이 (hip → foot) | ~0.55 m | ~0.34 m | **~0.24 m** |
| 액추에이터 | SEA, 토크 제어 | QDD BLDC, 토크 제어 | **취미용 서보, 위치 제어 only** |
| 최대 관절 토크 | ~80 N·m | ~17 N·m | **~1 N·m (MG996R 6V 기준)** |
| 토크/질량 비 | 2.67 N·m/kg | ~1.89 N·m/kg | **~0.67 N·m/kg** |
| 관절 응답 지연 | <5 ms | <5 ms | **20~50 ms** |
| 위치 분해능 | encoder, ~0.01° | encoder, ~0.01° | **포텐셔미터, ~0.5~1°** |
| 백래시 | 무시 가능 | 무시 가능 | **1~3° (감속비 의존)** |

**결론**: 논문의 보상 scale, PD gain, 명령 속도 범위, action_scale을 그대로 가져오면 시뮬레이션에서는 학습되어도 실제 로봇으로 옮길 수 없는 정책이 나온다. 더 나쁜 경우, 시뮬레이션 자체가 hobby servo로는 불가능한 동작을 학습해 ep_len이 학습 신호로 의미 없는 값이 된다.

---

## 1. Spot Micro 물리 모델 — 검증해야 할 항목

다음 값들은 **빌드마다 다르므로 실측 또는 CAD 확인이 필요하다**. AI 에이전트와 작업을 시작하기 전에 이 표를 먼저 채워야 한다.

### 1.1 측정 필요 항목

```
[ ] 총 질량 (배터리 포함):           _____ kg
[ ] 몸통 질량 (다리 제외):            _____ kg
[ ] 다리 1개 질량:                   _____ kg
[ ] 다리 길이 - upper (hip→knee):    _____ m
[ ] 다리 길이 - lower (knee→foot):   _____ m
[ ] 어깨 폭 (좌↔우 hip joint 거리):   _____ m
[ ] 골반 깊이 (앞↔뒤 hip joint 거리): _____ m
[ ] COM 높이 (서있을 때, 바닥 기준):   _____ m
[ ] 사용 서보 모델명:                  _____
[ ] 서보 정격 전압:                    _____ V
[ ] 서보 stall torque @ 정격:         _____ N·m
[ ] 서보 no-load speed @ 정격:        _____ rad/s
[ ] 배터리 종류 / 정격 전압:          _____
```

### 1.2 시뮬레이션 URDF/USD 검증 체크리스트

```
[ ] URDF의 link mass와 실측 질량 차이 < 10%
[ ] URDF의 link inertia가 box/cylinder 근사가 아닌 실제 형상 기반
[ ] joint limit이 서보 물리 한계와 일치 (현재: shoulder ±0.548, leg [-2.666,1.548], foot [-2.600,0.100])
[ ] joint friction이 비영(0이면 hobby servo 마찰 무시)
[ ] joint damping이 PD gain의 damping과 별개로 설정됨
```

특히 link inertia가 잘못되면 학습된 정책이 실제로는 발산한다. URDF에 `<inertia ixx="0.0001" .../>` 같은 placeholder가 남아있는지 확인할 것.

---

## 2. 논문 적용 시 깨지는 가정 — 항목별 재해석

### 2.1 Action scale (Rudin: 0.25)

**Rudin의 가정**: torque control이며 action은 PD target offset (rad). PD gain stiffness ~80 N·m/rad, damping ~2 N·m·s/rad이라 0.25 rad offset → 최대 20 N·m demand → ANYmal 가능.

**Spot Micro에 옮기면**: stiffness 60 × 0.25 = 15 N·m demand. 하지만 MG996R 한계 ~1 N·m → **시뮬레이션은 매 step 토크가 15× 과대 산정**. 정책은 이 "비현실적인 토크"에 의존하여 학습된다.

**물리 기반 재산정**:
```
desired action_scale × stiffness ≤ 0.7 × servo_stall_torque

여기서 0.7은 안전계수 (지속 가능한 토크는 stall의 70% 이하).

Spot Micro (MG996R, 6V) 가정:
  servo_stall_torque ≈ 1.0 N·m
  안전 토크 ≈ 0.7 N·m
  → action_scale × stiffness ≤ 0.7

  case A: stiffness=60 → action_scale ≤ 0.012  ← 현재 0.25는 20배 과대
  case B: stiffness=20 → action_scale ≤ 0.035
  case C: stiffness=10 → action_scale ≤ 0.07
```

**즉, stiffness 60 + action_scale 0.25 조합은 hobby servo 영역 밖이다.** 학습이 잘 되더라도 실기기에서는 servo가 saturate한다.

**대안**: 시뮬레이션에 `effort_limit`(또는 max_torque)을 0.7 N·m로 명시적으로 걸어 PD output을 클립한다. 이러면 action_scale 0.25를 유지해도 정책이 "토크 한계를 인식하면서" 학습한다.

```python
# Isaac Lab actuator config 예시
actuators={
    "spot_servo": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=60.0,
        damping=3.0,
        effort_limit=0.7,        # ← 추가 (실제 서보 한계)
        velocity_limit=6.0,      # ← 추가 (rad/s, 서보 no-load speed)
    ),
}
```

### 2.2 PD gain (15차 60/3.0)

**문제**: §2.1에서 본 것처럼 stiffness 60은 hobby servo가 낼 수 없는 응답을 시뮬레이션하게 한다. 그러나 **stiffness가 너무 낮으면(30 이하) 학습 초기 random policy에서 자세가 무너진다** (TRAINING_HISTORY §6 Bug 5).

**해결**: stiffness curriculum.

```python
# Phase 1 (학습 초기 ~ 1500 iter): stiffness 60, damping 3.0
#   → 자세 안정, 학습 속도 우선
# Phase 2 (1500 ~ 3000 iter):     stiffness 60 → 30 선형 감소
#   → 정책이 점진적으로 hobby servo 영역에 적응
# Phase 3 (3000 iter ~ 끝):        stiffness 30, damping 1.5 + effort_limit 0.7
#   → 실기기 배포 가능 영역에서 fine-tuning
```

이 아이디어는 `TRAINING_HISTORY.md` §12 교훈 4의 발상을 명시적 schedule로 구현한 것이다.

### 2.3 Termination penalty (FUTURE_DIRECTION의 -5)

**FUTURE_DIRECTION.md의 -5 권장값**도 물리에 맞춰 재검토할 수 있다.

**원리**: termination penalty는 다음을 만족해야 한다.
```
|termination|  <  alive × max_episode_length × 0.05
                  └─ advantage 분포 점유 방지
|termination|  >  per_step_alive × E[ep_len_random_policy]
                  └─ random policy가 일부러 죽는 것을 방지
```

Spot Micro에서 random policy의 평균 생존이 약 30 step이라 가정:
```
하한: 1.0 × 30 = 30  (random이 죽으려는 경향 차단)
상한: 1.0 × 1000 × 0.05 = 50  (advantage 점유 방지)

→ 권장: 30 ~ 50 사이.
→ FUTURE_DIRECTION의 -5는 advantage 점유는 막지만 random policy가 일부러 죽는 것을 막기엔 약할 수도 있다.
```

**수정 권장**: -5에서 시작해서, Phase 1 진단 로그에서 `diag/term_ratio`(종료 비율)가 학습 진행에도 떨어지지 않으면 -10, -20으로 단계적으로 키운다. **절대 -200으로 점프하지 않는다.**

### 2.4 명령 속도 범위 (Stage3: 0.2~0.6 m/s)

**Froude 수 기반 재산정**: 동역학적으로 동등한 보행 속도는 다음으로 scaling된다.
```
Fr = v² / (g × L_leg)

ANYmal 평지 보행 1 m/s:  Fr = 1² / (9.8 × 0.55) = 0.186
Mini Cheetah 1.5 m/s:    Fr = 1.5² / (9.8 × 0.34) = 0.675

Spot Micro (L_leg = 0.24 m)에서 동등 Fr:
  Fr=0.186 (안정 보행) →  v = √(0.186 × 9.8 × 0.24) = 0.66 m/s
  Fr=0.675 (질주)       →  v = √(0.675 × 9.8 × 0.24) = 1.26 m/s
```

**해석**: Spot Micro 0.6 m/s는 ANYmal의 1 m/s에 해당하는 *비교적 빠른 보행*이다. **hobby servo 한계(서보 속도, 토크)를 고려하면 실기기에서 안정적으로 낼 수 있는 속도는 약 0.2~0.4 m/s**가 현실적이다.

**권장**:
```
Stage2 (트롯):  0.05 ~ 0.20 m/s   (Fr ≤ 0.017, 매우 안정)
Stage3 (전속도): 0.10 ~ 0.35 m/s   (Fr ≤ 0.052, 학습 후 sim-to-real 전 추가 fine-tuning 필요)
```

`TRAINING_HISTORY.md`의 0.2~0.6 m/s는 시뮬레이션에서는 학습되어도 실기기 sim-to-real에서 거의 확실히 실패한다.

### 2.5 Foot clearance 6 cm (Margolis)

**Margolis의 가정**: Mini Cheetah의 다리 길이 ~0.34 m, 6 cm = 다리의 17.6%.

**Spot Micro 다리 길이 0.24 m에 비례 적용**:
```
clearance_target = 0.176 × 0.24 ≈ 4.2 cm
```

6 cm를 그대로 두면 swing foot가 다리 길이의 25%까지 올라가야 하는데, 이는 hobby servo의 빠른 응답이 필요해서 학습에 방해가 된다.

### 2.6 Action smoothing α=0.8 (EMA)

**Margolis의 가정**: 100 Hz 제어에서 α=0.8 → 시정수 ~50 ms.

**Spot Micro 50 Hz**: α=0.8이면 시정수 ~100 ms. **이미 hobby servo 응답 시간(~50 ms)보다 느려서 명령이 더 둔해진다.**

**물리 기반 권장**:
```
α를 시정수 기준으로 재계산:
  τ_smoothing = -dt / ln(1 - α)
  목표 시정수 = 30 ms (servo 응답 시간 미만)
  dt = 20 ms (50 Hz)
  → α = 1 - exp(-20/30) = 0.487  ≈ 0.5

따라서 EMA: processed = 0.5 × action + 0.5 × processed_prev
```

**그러나** 학습 초기에는 α=0.8이 jitter 억제에 더 효과적이다. → **smoothing curriculum**:
```
iter 0~1000:    α = 0.8 (강한 smoothing, 학습 안정)
iter 1000~3000: α = 0.8 → 0.5 선형
iter 3000+:     α = 0.5 (실기기 영역)
```

### 2.7 Sensor noise & delay (논문은 보통 안 다룸)

Rudin·Margolis는 Domain Randomization으로 해결하지만, hobby robot은 더 강한 처리가 필요하다.

| 신호 | 시뮬레이션 ground truth | 실기기 현실 | DR 권장 |
|------|---------------------|----------|---------|
| Joint position | exact, 0 delay | ±1° 분해능, ~10ms 지연 | noise σ=0.01 rad, delay 1 step |
| Joint velocity | exact (해석적 미분) | 위치 미분, 노이즈 큼 | noise σ=0.5 rad/s 또는 obs에서 제외 |
| Base orientation (gravity_b) | exact | IMU 노이즈, ~5ms 지연 | noise σ=0.05 (normalized vector) |
| Base linear velocity | exact | **추정 불가** (real에서 얻을 수 없음) | obs에서 **제거**해야 sim-to-real 가능 |
| Base angular velocity | exact | gyro, 노이즈·바이어스 있음 | noise σ=0.2 rad/s, bias 추가 |
| Foot contact | exact boolean | **센서 없음** (대부분 hobby quad) | 추정으로 대체 또는 obs에서 제외 |

**가장 큰 함정**: `base_lin_vel`을 observation에 포함하면 시뮬레이션에서 학습은 잘 되지만 실기기에는 절대 옮길 수 없다. hobby quad는 SLAM·외부 모션 캡처 없이 base velocity를 얻을 방법이 없다.

→ **observation에 포함되어 있는지 즉시 확인 필요**. 포함되어 있다면 *현재까지의 학습은 sim-to-real이 불가능한 정책*이다.

---

## 3. 물리 기반 파라미터 통합 표

`FUTURE_DIRECTION.md`의 권장값을 §2의 물리 분석으로 보정한 최종 권장값.

### 3.1 Actuator (Phase 1 즉시 적용)

```python
# stiffness curriculum 적용 전 baseline
actuators = {
    "spot_servo": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=60.0,           # Phase 1: 학습 안정 우선
        damping=3.0,
        effort_limit=0.7,         # ← 신규 (서보 stall × 0.7)
        velocity_limit=6.0,       # ← 신규 (서보 no-load speed)
    ),
}

# action_scale은 그대로 두되 effort_limit으로 클립
action_scale = 0.25
```

### 3.2 보상 (Stance, 물리 보정 후)

| 보상 | FUTURE_DIRECTION | 물리 보정 후 | 보정 사유 |
|------|------------------|-------------|----------|
| alive | +1.0 | +1.0 | 유지 |
| upright | +1.5 (sigma 0.1) | +1.5 (sigma 0.1) | 유지 |
| gravity | -0.5 | -0.5 | 유지 |
| lin_vel_xy | -0.05 | -0.05 | 유지 |
| ang_vel_z | -0.05 | -0.05 | 유지 |
| joint_default | -0.1 | -0.1 | 유지 |
| foot_spread | -0.2 | -0.2 | 유지 |
| foot_slip | -0.02 | -0.02 | 유지 |
| stand_still | 0.0 | 0.0 | 유지 |
| **termination** | -5.0 | **-30.0** | random policy ep_len ~30 step 기반 |
| **action_acc** | (Phase 2) | **-0.001** ← Phase 2 추가 시 | hobby servo bandwidth 보호 |
| **torque_penalty** | 미정 | **-0.0005** ← Phase 2 추가 시 | sum(|tau|) 패널티, servo 발열 방지 |

### 3.3 명령 속도 범위 (Stage 별)

```python
# Stage 2 (Trot)
cmd_lin_vel_x: (0.05, 0.20)    # m/s, Fr ≤ 0.017
cmd_lin_vel_y: (-0.05, 0.05)   # m/s, lateral
cmd_ang_vel_z: (-0.3, 0.3)     # rad/s

# Stage 3 (Direct)
cmd_lin_vel_x: (0.10, 0.35)    # m/s, Fr ≤ 0.052
cmd_lin_vel_y: (-0.10, 0.10)   # m/s
cmd_ang_vel_z: (-0.5, 0.5)     # rad/s
```

### 3.4 Foot clearance / smoothing

```python
foot_clearance_target = 0.042  # 4.2 cm (다리 길이의 17.6%)

# EMA smoothing curriculum
def get_ema_alpha(iter, total=5000):
    if iter < 1000:
        return 0.8
    if iter < 3000:
        return 0.8 - 0.3 * (iter - 1000) / 2000
    return 0.5
```

### 3.5 Domain Randomization (Phase 3)

```python
# 학습 후 sim-to-real 직전에 활성화
domain_randomization = {
    # 질량
    "mass_range":         (0.85, 1.15),     # ±15%
    "com_offset_xy":      (-0.02, 0.02),    # m
    "com_offset_z":       (-0.01, 0.01),    # m
    
    # 액추에이터
    "stiffness_range":    (0.8, 1.2),       # ±20%
    "damping_range":      (0.7, 1.3),       # ±30% (hobby servo 마찰 다양)
    "effort_limit_range": (0.85, 1.15),     # 서보 개체차
    "action_delay_steps": (0, 2),           # 0 ~ 40 ms
    
    # 환경
    "ground_friction":    (0.4, 1.2),       # 매끄러운 바닥 ~ 카펫
    "ground_restitution": (0.0, 0.3),
    
    # 센서
    "imu_noise_std":      0.05,             # gravity_b 정규화 벡터
    "joint_pos_noise":    0.01,             # rad
    "joint_pos_delay":    1,                # step (20 ms)
}
```

---

## 4. AI 에이전트 협업 프로토콜

AI 에이전트(Claude Code, Cursor 등)와 작업할 때 구조화된 프로토콜이 없으면 다음 문제가 발생한다:
- 에이전트가 한 번에 여러 파라미터를 바꾸어 §12 교훈 1을 위반
- 학습 결과 해석을 에이전트에게 의존하다 잘못된 진단으로 다음 단계 진행
- "잘 됩니다" 같은 모호한 보고로 회귀가 누적

### 4.1 단일 변경 원칙 (Single-Variable Rule)

**모든 학습 세션은 다음 형식으로 시작한다**:

```yaml
# session_NN.yaml
session_id: 19
date: 2026-04-28
hypothesis: "termination -200 → -30이 ep_len 회귀의 원인이다"
predicted_outcome: "ep_len이 1000 iter 안에 ≥ 100으로 회복"
falsification: "ep_len이 1000 iter 후에도 < 80이면 가설 기각"

baseline_commit: 4bc87f7    # 18차 마지막 commit
changes:
  - file: stance_cfg.py
    line_or_field: rew_scale_termination
    from: -200.0
    to: -30.0
    reason: "FUTURE_DIRECTION §1.1 + PHYSICS §2.3"
# 다른 변경은 없어야 함. 있으면 본 세션 무효.
```

**에이전트 행동 규칙**:
1. `changes` 항목이 2개 이상이면 에이전트는 수정 작업을 거부하고 사용자에게 분리를 제안한다.
2. 각 세션 완료 시 다음 형식으로 결과를 기록한다.

```yaml
# session_NN_result.yaml
session_id: 19
duration_iter: 1000
final_metrics:
  ep_len_mean: ____
  mean_reward: ____
  entropy: ____
  diag/per_step_net: ____
  diag/term_ratio: ____
hypothesis_outcome: confirmed | refuted | inconclusive
next_action: "..."
```

### 4.2 에이전트 행동 결정 트리

학습 종료 후 에이전트는 다음 트리에 따라 다음 step을 *제안*한다 (실행 결정은 인간).

```
1. ep_len이 회복(≥ 100)되었는가?
   ├─ YES → Phase 2 진입 (한 번에 한 보상 항목 추가)
   └─ NO  → 2로
   
2. entropy가 학습 중 감소했는가?
   ├─ YES → Phase 1 적용했으나 다른 원인. H2/H3 검증으로 진행
   └─ NO (증가) → advantage signal 여전히 약함
                 → termination을 -30 → -10으로 약화 시도
                 → 또는 Adam learning rate 1e-3 → 5e-4 감소

3. diag/per_step_net이 양수인가?
   ├─ YES → 보상 구조는 OK, 다른 원인 (관찰값 노이즈, reset 버그)
   └─ NO  → stance reward overconstrain (H2). foot_spread, joint_default 추가 약화

4. diag/term_ratio (종료 비율)가 학습 중 감소했는가?
   ├─ YES → 정책이 균형을 학습 중, 그저 시간이 더 필요
   └─ NO  → termination 조건이 지나치게 strict하거나 PD gain 부족
```

이 트리는 코드 파일로 두어 에이전트가 *언제든* 참조하도록 한다 (예: `agent_decision_tree.md`).

### 4.3 에이전트 금지 사항

다음 행동을 에이전트가 시도하면 사용자가 즉시 중단시킨다:

- ❌ 학습 결과를 보지 않고 "이렇게 하면 잘 될 겁니다"라며 코드 수정 (이론 우선의 함정)
- ❌ 텐서보드 로그 없이 ep_len 숫자 하나만으로 진단
- ❌ 한 번에 2개 이상 파라미터 수정
- ❌ "Rudin/Margolis 논문에서는 이 값을 씁니다"를 *유일한* 근거로 제시 (이 문서 §2 전체가 그것이 깨지는 이유다)
- ❌ 학습 도중에 코드 수정 (학습 종료 후 다음 세션에서 변경)

### 4.4 에이전트가 매 세션 시작 시 수행할 절차

```
1. 이전 세션의 session_NN_result.yaml을 읽고 요약
2. 현재 git commit과 baseline_commit이 일치하는지 확인
3. session_NN+1.yaml의 hypothesis와 changes를 사용자와 합의
4. 변경 적용 (단일 파라미터 원칙 준수)
5. git commit (메시지 형식: "session 19: termination -200 → -30")
6. 학습 시작
7. 학습 중에는 코드 수정 금지. 텐서보드만 모니터링
8. 학습 종료 후 result.yaml 작성, 결정 트리에 따라 다음 제안
```

### 4.5 에이전트와의 의사소통 템플릿

**좋은 질문 (에이전트에게 던질 만한)**:
- "session_19의 텐서보드 로그를 분석하고, FUTURE_DIRECTION §6 KPI 대비 충족/미충족 항목을 표로 정리해줘"
- "결정 트리 §4.2 step 2에서 'YES'에 해당하는데, H2/H3 중 어느 것이 더 가능성 높은지 코드를 직접 확인해서 답해줘"
- "현재 effort_limit이 설정되어 있는지 actuator config를 view로 확인해줘"

**나쁜 질문**:
- "어떻게 하면 ep_len이 올라갈까?" (에이전트가 추측으로 답함, 변수 동시 변경 위험)
- "이 코드 좀 고쳐줘" (어떤 가설을 검증하는지 불명)

---

## 5. Validation Gate — 단계 전환 검증

각 Phase 종료 시 다음 모든 항목이 통과해야 다음 Phase로 진행한다. 하나라도 실패하면 해당 Phase에 머무른다.

### 5.1 Phase 1 → Phase 2 Gate

```
[ ] Stance ep_len ≥ 200 @ 1000 iter
[ ] entropy 학습 중 감소 추세
[ ] diag/per_step_net > 0 평균
[ ] diag/term_ratio < 30% @ 마지막 100 iter
[ ] base_lin_vel이 observation에 포함되어 있지 않음 (sim-to-real 가능성 보존)
[ ] effort_limit이 actuator config에 설정됨
[ ] reset 시 EMA buffer 초기화 확인됨
```

### 5.2 Phase 2 → Phase 3 Gate

```
[ ] Stance ep_len ≥ 800 @ 5000 iter
[ ] random push 1N 외란에 대해 5초 회복
[ ] action_acc 평균 < 5 rad/s² (jitter 없음)
[ ] sum(|joint_torque|) 평균 < 4 N·m (서보 발열 안전 영역)
[ ] 시각적으로 자연스러운 자세 (영상 5초 관찰)
[ ] stiffness curriculum 끝 값(30, 1.5)에서도 ep_len ≥ 600
```

### 5.3 Phase 3 → Sim-to-Real Gate

```
[ ] Stage2 트롯 ep_len ≥ 800 @ 0.10 m/s
[ ] Stage3 ep_len ≥ 600 @ 0.25 m/s
[ ] DR 활성화 후에도 위 조건 유지
[ ] sum(|joint_velocity|) 평균 < 24 rad/s (4다리 × 평균 6 rad/s, no-load 미만)
[ ] 사용된 토크 분포에서 95th percentile < 0.7 N·m (effort_limit 이내)
[ ] policy를 ONNX/TorchScript로 export 가능
```

### 5.4 Sim-to-Real Gate

```
[ ] real robot에서 5초 자립 (정지 명령)
[ ] real robot에서 0.1 m/s 보행 3초 유지
[ ] 서보 온도 안정 (5분 동작 후 70°C 미만)
[ ] 배터리 1회 충전으로 5분 이상 동작
```

---

## 6. 첫 세션 (session_19) 명세

이 문서를 적용한 첫 세션을 다음과 같이 정의한다. 에이전트는 이 명세를 그대로 실행한다.

```yaml
session_id: 19
date: TBD
hypothesis: |
  현재 termination -200이 PPO advantage 분포를 점유해 다른 보상 신호를
  노이즈 수준으로 약화시킨다. -30으로 정정하면 ep_len이 회복된다.

predicted_outcome: |
  - ep_len_mean이 1000 iter 안에 ≥ 100으로 회복
  - entropy가 학습 중 감소 추세 전환
  - diag/per_step_net 평균 > 0

falsification: |
  1000 iter 후에도 ep_len < 80이거나 entropy가 여전히 증가하면 가설 기각.
  → H2(stance overconstrain) 검증으로 전환.

baseline_commit: 4bc87f7

changes:
  # 단일 변경 원칙: 본질적 1개 + 진단 로깅 1개 = 2개 묶음 허용
  - file: stance_cfg.py
    field: rew_scale_termination
    from: -200.0
    to: -30.0
    reason: "PHYSICS §2.3 (random ep_len 30 step 기반)"
  
  - file: env.py
    function: _get_rewards
    change: "self.extras['log']에 항목별 보상 + 진단 메트릭 추가"
    reason: "FUTURE_DIRECTION §5.2 진단 로깅"

verification_before_run:
  - "git diff를 출력하여 위 2개 파일 외 변경이 없는지 확인"
  - "actuator config에서 effort_limit 존재 확인 (없으면 별도 세션 20에서 추가)"
  - "observation에 base_lin_vel이 포함되어 있는지 확인 → result.yaml에 기록"

duration: 1000 iter
```

---

## 7. 결론

`FUTURE_DIRECTION.md`의 단일 파라미터 정정(termination -200 → -5)이 정확한 첫걸음이지만, 이 문서는 그것조차 **Spot Micro의 물리에 맞춰 -30 정도로 보정**해야 함을 보였다. 더 본질적으로:

1. **action_scale 0.25 + stiffness 60은 hobby servo에서 불가능한 토크를 요구한다.** `effort_limit=0.7`을 actuator config에 추가하지 않으면 시뮬레이션에서 학습된 정책은 sim-to-real이 원천적으로 막혀 있다.
2. **observation에 `base_lin_vel`이 있는지 확인하라.** 있으면 지금까지의 모든 학습은 실기기에 옮길 수 없는 정책이다.
3. **명령 속도 0.6 m/s는 Spot Micro에는 ANYmal의 1 m/s에 해당하는 빠른 보행이다.** Stage3는 0.35 m/s로 낮춘다.
4. **AI 에이전트와 협업 시 단일 변경 원칙을 코드로 강제하라.** session_NN.yaml + result.yaml 페어로 변경 이력을 강제 추적한다.

다음 행동은 §6의 session_19 명세 그대로다. 이 세션 결과로 가설이 확정되면 Phase 2로, 기각되면 §4.2 결정 트리 step 2~4로 분기한다.
