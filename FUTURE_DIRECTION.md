# Spot Micro 강화학습 — 향후 방향성 제안

> 작성일: 2026-04-27
> 기반 문서: `TRAINING_HISTORY.md` (1~18차 학습 기록)
> 목적: 18차 학습 정체 원인의 재진단, 관련 연구 정리, 단계별 실행 계획

---

## 0. TL;DR (요약)

**진단**:
1. "**15차 ep_len 212**"는 성공이 아니라 `clamp(min=0)`이 만든 *수치적 환영(numerical illusion)*이다. 실제 학습된 정책은 stationary survival에 가까웠고, 이후 단계가 실패한 것은 자연스러운 결과다.
2. "**Rudin 표준 -200**"은 잘못된 인용이다. Rudin et al. 2021/2022 원본의 termination penalty는 약 **-2.0** 수준이며, -200은 reward scale이 훨씬 큰 다른 연구의 값이거나 typo일 가능성이 높다. 18차까지의 디버깅이 그릇된 baseline 위에서 진행되었다.
3. 18차의 ep_len 역행은 **early death trap**이 아니라 **PPO advantage signal degradation**이 더 정확한 설명이다 — termination -200이 단일 transition의 advantage를 점유해 다른 보상 신호가 노이즈 수준으로 묻힌다.
4. **Entropy가 학습 중 *증가*하는 패턴(11.08 → 15.34)**은 정책이 수렴하지 못하고 오히려 random 쪽으로 발산한다는 신호다.

**권장 방향**:
- **Phase 1 (1주)**: 진단 로깅 추가 + termination penalty를 −200 → −2~−5 로 정정 + 한 번에 한 변수만 바꾸는 이분 탐색 시작
- **Phase 2 (2~3주)**: positive-dominant reward 구조 재설계 (alive·tracking이 모든 패널티 합을 압도하도록)
- **Phase 3 (1개월+)**: reward curriculum + domain randomization 도입, sim-to-real 준비

---

## 1. 18차 결과의 재해석 — "early death trap"을 넘어서

`TRAINING_HISTORY.md` §6 Bug 1은 early death trap을 다음 부등식으로 정의했다:
```
sum(per_step_penalty) × ep_len > |termination_penalty|
```

이 공식 자체는 옳지만, **18차에서 stiffness 60 복원 + `living.clamp(min=0) + rew_termination` 절충을 적용했음에도 ep_len이 역행한다**는 사실을 설명하지 못한다. 절충 구조에서는 living은 항상 ≥ 0이므로 위 부등식은 성립할 수 없다.

### 1.1 새로운 진단: PPO Advantage Signal Degradation

PPO는 advantage `A(s,a) = Q(s,a) − V(s)`로 policy gradient를 계산한다. RSL-RL은 GAE(λ) 후 advantage normalization (mean=0, std=1)을 적용한다.

문제 시나리오:
```
4096 envs × rollout 24 step ≈ 98k transitions / iteration
ep_len 40 가정 시 종료 transition ≈ 24/40 × 4096 ≈ 2,460개 (~2.5%)

종료 transition 1개의 advantage 기여:
  raw return = (생존 step의 작은 양수) + (-200)
            ≈ 0.5 × 24 - 200 ≈ -188

비종료 transition의 raw advantage ≈ ±1~5 수준
→ std가 -200 한 방으로 폭증
→ 정규화 후 비종료 신호의 advantage가 ~0.01 수준으로 묻힘
```

**즉, termination penalty가 너무 크면 다른 보상의 학습 신호가 노이즈 수준으로 약화된다.** 절충 구조도 이 문제를 해결하지 못한다 — 종료 시점의 -200은 그대로 advantage에 들어가기 때문이다.

### 1.2 Entropy 증가 패턴이 보여주는 것

| 차수 | iter | entropy | action_std |
|------|------|---------|------------|
| 16차 | 1 | 11.08 | 0.61 |
| 16차 | 205 | **13.87** | 0.79 |
| 17차 | 152 | **15.09** | 0.85 |
| 18차 | 125 | **15.34** | 0.87 |

학습이 진행되면서 entropy와 action_std가 *증가*한다는 것은 policy가 deterministic으로 수렴하지 못하고 random 방향으로 발산한다는 의미다. 정상적으로 학습되는 PPO에서는 entropy가 entropy_coef에 따라 천천히 *감소*한다. 이는 §1.1과 정합적이다 — gradient 신호가 너무 약해 entropy 항이 dominant해진다.

### 1.3 15차 "성공"의 재해석

`return total.clamp(min=0.0)`은 다음을 의미한다:
```
모든 음수 보상의 gradient = 0 (clamp의 backward는 음수 영역에서 0)
→ 학습되는 보상은 alive(1.0) + upright(exp tracking) + (보조 양수항)
→ 사실상 "alive bonus + upright tracking"만으로 학습
```

ep_len 212는 1000 step 만점의 21%다. 이는 robust standing이라기보다 **"넘어지지 않을 정도로만 자세를 잡는 minimum-effort policy"**다. Stage2(트롯)에서 실패한 이유는 이 정책이 실제로는 동적 균형이 없는 정적 자세였기 때문이다.

→ **15차로 회귀(`TRAINING_HISTORY.md` Option A)는 권장하지 않는다.** 같은 local optimum에 다시 갇힐 뿐이다.

---

## 2. 외부 연구 정리 — 무엇이 표준인가

### 2.1 Rudin et al. 2021 — "Learning to Walk in Minutes" (CoRL)

- **시뮬레이터**: Isaac Gym, 4096 envs (이 프로젝트와 동일)
- **termination penalty**: 코드 기준 약 **-1.0 ~ -2.0** (`legged_gym/.../rewards.py`)
- **alive bonus**: 명시적으로 없음. positive shaping은 *tracking* 보상으로 대체
- **핵심 보상 구조**:
  ```
  tracking_lin_vel  (+1.0,  exp(-error²/0.25))
  tracking_ang_vel  (+0.5,  exp(-error²/0.25))
  feet_air_time    (+1.0,  swing time bonus)
  termination       (-2.0)
  
  # smoothing/regularization (모두 작은 음수)
  lin_vel_z         (-2.0)
  ang_vel_xy        (-0.05)
  orientation       (-0.0)
  torques           (-0.0001)
  dof_acc           (-2.5e-7)
  action_rate       (-0.01)
  ```

**중요**: 양수 보상이 tracking 위주로 매 step 0.5~2.0의 신호를 제공하고, termination은 -2.0으로 다른 신호를 묻지 않을 수준이다. **현재 프로젝트의 -200은 이보다 100배 강하다.**

### 2.2 Margolis & Agrawal 2023 — "Walk These Ways" (CoRL)

- **multiplicity of behaviors**: 명령에 따라 다양한 gait 학습
- **foot clearance**: -2.0 (현재 프로젝트와 동일)
- **foot slip**: -0.04 (현재는 -0.02 → 적정)
- **action smoothing**: first-order low-pass (현재의 EMA와 유사)
- **termination penalty**: 명시적으로 강조하지 않음 (즉, 작은 값)

`TRAINING_HISTORY.md` §5.2에서 인용한 Margolis 표준은 보상 항목 구성은 맞지만, **scale 조정에 termination penalty 정상화가 누락**되어 있다.

### 2.3 Cheng et al. 2024 — "Extreme Parkour with Legged Robots"

- two-stage curriculum: flat → terrain
- reward curriculum: 학습 초기 *soft* reward → 후기 *hard* reward 전환
- termination: terrain failure만 강한 패널티, 일반 fall은 작은 값
- **시사점**: hobby quadruped에서도 reward를 단계적으로 강화하는 schedule이 유효

### 2.4 Zhuang et al. 2023 — "Robot Parkour Learning"

- pre-training: 단순 보상으로 기본 보행 습득
- fine-tuning: parkour-specific reward 추가
- **시사점**: 이 프로젝트의 Stance → Trot → Direct 커리큘럼은 이론적으로 옳다. 다만 *Stance 단계의 보상이 너무 복잡*하다.

### 2.5 종합

학계 표준은 다음과 같이 요약된다:

| 항목 | 학계 표준 | 현재 프로젝트 | 평가 |
|------|----------|--------------|------|
| termination penalty | -1 ~ -10 | **-200** | ❌ 100× 과대 |
| alive bonus | 0 ~ 1.0 (선택) | +1.0 | ✓ |
| tracking reward 합 | +1.5 ~ +3.0 | (Stance에서) +1.5 (alive+upright) | △ |
| 패널티 항목 합 | -0.1 ~ -0.5 (per step) | -1.0 ~ -3.0 | △ 다소 강함 |
| action smoothing | EMA 0.7~0.9 | 0.8 | ✓ |
| PD gain (hobby) | stiffness 20~50, damping 0.5~2.0 | 60/3.0 | △ 약간 강함, but 학습 안정 위해 OK |

**결론: termination penalty 1개만 정상화해도 18차의 정체가 풀릴 가능성이 매우 높다.**

---

## 3. 근본 원인 가설 (우선순위 재정렬)

### 🔴 H1. Termination penalty −200이 advantage 분포를 점유

§1.1에 상술. 단일 transition의 advantage가 다른 모든 신호를 dominate.

**검증 방법**: termination을 -2로 낮추고 다른 모든 변수 고정 → ep_len 추이 관찰. 1000 iter 안에 결론 가능.

**예측**: ep_len이 ≥ 100으로 빠르게 회복.

### 🔴 H2. Stance reward 구조가 "걷기 금지"를 너무 강하게 명령

현재 stance에는 `lin_vel_xy(-0.1) + ang_vel_z(-0.1) + stand_still(-0.1) + foot_spread(-0.5) + joint_default(-0.2) + foot_slip(-0.02) + gravity(-2.0×tilt)`로 이동을 적극 억제한다. 하지만 stance의 목적은 "동적 균형 유지"이지 "완전 정지"가 아니다.

학습 초기 random policy는 작은 이동·회전이 불가피한데, 이걸 모두 패널티로 막으니 *아무 행동도 하지 않는 게 최적*이 된다. 그러나 random init torque로 인해 자세가 흔들리고, 결국 termination → ep_len 역행.

**검증 방법**: stand_still 한 항목만 0으로 두고 학습.

### 🟡 H3. EMA action smoothing의 reset 시 초기화 누락 (의심)

```python
# _reset_idx에서 self.processed_actions[env_ids] = 0 같은 처리가 있는가?
# 없다면 reset 직후 첫 step에서:
#   processed_new = 0.8 × action_new + 0.2 × processed_old(이전 episode)
# → 이전 episode 끝에서 발산하던 action이 새 episode 첫 step에 영향
```

**검증 방법**: env.py의 `_reset_idx` 코드를 확인하여 `processed_actions`, `_last_actions`, `_last_last_actions`, `_last_joint_vel`이 모두 reset 시 0으로 초기화되는지 점검.

### 🟡 H4. compute_rewards 내 호출 순서 (TRAINING_HISTORY §9 문제 3)

저자가 이미 의심한 부분. Isaac Lab `DirectRLEnv.step` 표준 순서는:

```
pre_physics_step() → physics × decimation → _apply_action()
→ _get_dones() → _get_rewards() → _reset_idx() → _get_observations()
```

이 순서라면 `_get_rewards` 시점의 `self._last_actions`는 *이번 step의 action*이고, 이전 step의 action은 별도 buffer가 필요하다. action_acc(2차 미분) 계산이 의도와 다를 수 있다.

**검증 방법**: 디버그 print로 `(self.actions - self._last_actions).abs().max()`가 매 step 0이 아닌지 확인.

### 🟢 H5. PD gain 60/3.0이 hobby servo 한계 초과

학습은 잘 되지만 sim-to-real 시 servo가 명령을 못 따라가서 실패할 수 있다. 학습 자체의 문제는 아니므로 우선순위 낮음.

---

## 4. 권장 실행 계획

### Phase 1 — 즉시 실행 (1주)

#### Step 1.1: 진단 로깅 강화

`_get_rewards` 끝에 항목별 통계를 텐서보드에 기록한다.

```python
# rsl_rl extras에 추가
self.extras["rew/alive"]         = rew_alive.mean().item()
self.extras["rew/upright"]       = rew_upright.mean().item()
self.extras["rew/gravity"]       = rew_gravity.mean().item()
self.extras["rew/foot_slip"]     = rew_foot_slip.mean().item()
self.extras["rew/termination"]   = rew_termination.mean().item()
self.extras["rew/per_step_net"]  = (total_excl_term).mean().item()
self.extras["diag/term_height"]  = (height_terminated).float().mean().item()
self.extras["diag/term_tilt"]    = (tilt_terminated).float().mean().item()
self.extras["diag/term_ratio"]   = (self.reset_buf).float().mean().item()
```

**왜 중요**: per_step_net이 양수인지 음수인지를 모르면 모든 가설이 추측에 불과하다. 한 번의 학습으로 H1, H2의 진위를 동시에 확인할 수 있다.

#### Step 1.2: Termination penalty 정상화

```python
# stance_cfg.py
rew_scale_termination: -200.0 → -5.0
```

`-5.0`은 alive(+1.0) × max_ep_len(1000) = 1000 대비 0.5%로, advantage 분포를 점유하지 않는다. 동시에 alive bonus만으로는 메우기 힘든 의미 있는 패널티다.

다른 모든 변수는 18차 그대로 유지한다. **한 번에 한 변수만 바꾼다는 §12 교훈 1을 엄격히 준수한다.**

#### Step 1.3: Reset 초기화 점검

`env.py` `_reset_idx`에서 다음이 모두 0으로 초기화되는지 확인:
```
self.processed_actions[env_ids] = 0
self._last_actions[env_ids]      = 0
self._last_last_actions[env_ids] = 0
self._last_joint_vel[env_ids]    = 0
```

누락되어 있으면 추가. 이는 H3 검증.

#### Step 1.4: 1000 iter 학습 → 결과 분석

- ep_len이 ≥ 100으로 회복되면 H1 확정 → Phase 2로 진행
- 회복되지 않으면 진단 로그를 보고 H2/H3/H4 중 어느 것이 원인인지 판단

### Phase 2 — Reward 구조 재설계 (2~3주)

Phase 1에서 ep_len이 회복된 것을 확인한 다음에만 진행한다.

#### Step 2.1: Positive-dominant 보상 구조 명시화

```python
# 설계 원칙
# (1) per-step net reward 기댓값 ≥ +0.5 (random policy에서도 양수)
# (2) 모든 negative shaping 합 < 0.3 × positive 합
# (3) termination penalty | × max_ep_len 의 5% 이내
```

stance_cfg 권장 값 (Phase 1 결과 보고 미세조정):

| 보상 | 현재 | 권장 | 비고 |
|------|------|------|------|
| alive | +1.0 | +1.0 | 유지 |
| upright | +1.0 | +1.5 | sigma 0.04 → 0.1 (덜 sharp하게) |
| gravity | -2.0 | -0.5 | 약화 |
| lin_vel_xy | -0.1 | -0.05 | 약화 |
| ang_vel_z | -0.1 | -0.05 | 약화 |
| joint_default | -0.2 | -0.1 | 약화 |
| foot_spread | -0.5 | -0.2 | 약화 |
| foot_slip | -0.02 | -0.02 | 유지 |
| stand_still | -0.1 | **0.0** | 제거 (H2) |
| termination | (-5.0) | (-5.0) | Phase 1 적용값 |

#### Step 2.2: Stage2/3 cfg 동기화

`TRAINING_HISTORY.md` §4.1에서 지적된 `trot_cfg.py`의 누락된 보상 항목들을 추가한다. 단, scale은 Phase 2.1 원칙 준수.

#### Step 2.3: 이분 탐색 추가 변경

다음을 *순서대로* 한 번에 하나씩 추가하며 ep_len 영향 측정:
1. action_acc penalty 활성화 (-0.005)
2. dof_acc penalty 활성화 (-2.5e-7)
3. EMA smoothing α 미세조정 (0.8 → 0.7)

각 단계마다 500~1000 iter 학습. ep_len이 10% 이상 떨어지면 그 변경을 되돌린다.

### Phase 3 — Curriculum & Sim-to-Real 준비 (1개월+)

#### Step 3.1: Reward Curriculum

학습 진행도(예: iteration / total_iterations)에 따라 패널티 weight를 점진적으로 강화한다.

```python
def get_curriculum_scale(iteration, max_iter=5000):
    # 0 → 1로 천천히 증가, 1500 iter부터 본격 적용
    return min(1.0, max(0.0, (iteration - 1500) / 2000))

# 적용
rew_gravity_scale = -0.5 * get_curriculum_scale(iteration)
rew_foot_slip_scale = -0.02 * get_curriculum_scale(iteration)
```

학습 초기에는 alive + upright만 활성화 → 자세 잡히면 패널티 도입. Cheng 2024의 *soft → hard* 전략.

#### Step 3.2: Domain Randomization

sim-to-real 격차를 줄이기 위해 다음을 randomize:
- Mass: ±20% (link별)
- COM offset: ±2cm
- PD gain: ±20%
- Action delay: 0~2 step
- Joint friction: 0.0~0.05

#### Step 3.3: Stage2 트롯 학습 재개

Phase 2가 안정되면 Stage1 체크포인트를 로드하여 Stage2 시작. cmd 범위는 작게(0.1~0.2 m/s) 시작.

#### Step 3.4: Sim-to-real 단계

- Real Spot Micro 캘리브레이션 (servo calibration, IMU bias)
- Action smoothing α를 0.7 → 0.5로 더 강하게 (real servo는 더 느림)
- Emergency stop 로직

---

## 5. 즉시 실행할 코드 변경 (Phase 1 패치)

### 5.1 `stance_cfg.py` (또는 해당 base cfg)

```python
# 변경 전
rew_scale_termination: float = -200.0

# 변경 후
rew_scale_termination: float = -5.0
```

### 5.2 `env.py` `_get_rewards` 끝부분 추가

```python
# 진단 로깅 (Phase 1 후 제거 가능)
with torch.no_grad():
    self.extras["log"] = self.extras.get("log", {})
    self.extras["log"]["rew/alive"]        = rew_alive.mean().item()
    self.extras["log"]["rew/upright"]      = rew_upright.mean().item()
    self.extras["log"]["rew/gravity"]      = rew_gravity.mean().item()
    self.extras["log"]["rew/foot_slip"]    = rew_foot_slip.mean().item()
    self.extras["log"]["rew/joint_default"]= rew_joint_default.mean().item()
    self.extras["log"]["rew/foot_spread"]  = rew_foot_spread.mean().item()
    self.extras["log"]["rew/stand_still"]  = rew_stand_still.mean().item()
    self.extras["log"]["rew/termination"]  = rew_termination.mean().item()
    
    per_step_net = (
        rew_alive + rew_upright + rew_gravity + rew_foot_slip
        + rew_joint_default + rew_foot_spread + rew_stand_still
    )
    self.extras["log"]["diag/per_step_net"] = per_step_net.mean().item()
    self.extras["log"]["diag/per_step_neg_only"] = (
        per_step_net.clamp(max=0.0).mean().item()
    )
```

### 5.3 `env.py` `_reset_idx` 점검 (해당 buffer 존재 시)

```python
# 추가 (이미 있으면 OK)
if hasattr(self, "processed_actions"):
    self.processed_actions[env_ids] = 0.0
if hasattr(self, "_last_actions"):
    self._last_actions[env_ids] = 0.0
if hasattr(self, "_last_last_actions"):
    self._last_last_actions[env_ids] = 0.0
if hasattr(self, "_last_joint_vel"):
    self._last_joint_vel[env_ids] = 0.0
```

### 5.4 학습 후 텐서보드 확인 항목

- `rew/per_step_net` 평균 ≥ +0.5 (학습 100 iter 이후)
- `rew/termination` 평균이 다른 항목 합의 5배 이내
- `entropy`가 단조 감소(완만하게)
- `ep_len_mean`이 100 이후 단조 증가, 1000 iter에서 ≥ 200 목표

---

## 6. 성공 기준 (KPI)

### Phase 1 종료 조건
- [ ] Stance ep_len ≥ 200 @ 1000 iter (15차 재현)
- [ ] entropy 학습 중 감소 (현재 증가 중)
- [ ] per_step_net mean > 0
- [ ] termination ratio < 30% @ 1000 iter

### Phase 2 종료 조건
- [ ] Stance ep_len ≥ 800 @ 5000 iter (1000 step의 80%)
- [ ] random push 1N 견딤 (강건성)
- [ ] action 진동(dof_acc) 시각적으로 자연스러움

### Phase 3 종료 조건
- [ ] Stage2 트롯 ep_len ≥ 800 @ 0.2 m/s
- [ ] Stage3 전속도 ep_len ≥ 600 @ 0.4 m/s
- [ ] Real robot 5초 이상 자립

---

## 7. 위험 요소와 대응

| 위험 | 대응 |
|------|------|
| Phase 1에서 termination -5로 낮춰도 ep_len 회복 안 됨 | H2(stance reward overconstrain) 우선 검증. stand_still=0, foot_spread=-0.1 추가 약화 |
| Phase 2 적용 후 sliding policy 재발 | foot_slip을 -0.02 → -0.05로 강화, no_air 패널티 stance에도 약하게(-0.2) 활성화 |
| EMA smoothing이 학습을 방해 | α=0.8 → 0.5로 약화 또는 α=1.0 (무효화) 후 재시도 |
| stiffness 60이 sim-to-real에서 servo 한계 초과 | 학습 후반에 stiffness curriculum (60 → 40) 도입 후 fine-tuning |

---

## 8. 결론

18차까지의 디버깅은 **잘못된 baseline (Rudin -200) 위에서 진행된 미세조정**이었다는 점이 가장 큰 문제다. 보상 항목 구성(Margolis 2022)과 PD gain 진단(15차 vs 18차) 자체는 옳았으나, **단일 파라미터 정정(termination -200 → -5)이 빠져 있었다**.

권장하는 다음 행동은 명확하다:

1. **이번 주 안에**: termination penalty 정상화 + 진단 로깅 추가 + 1000 iter 학습
2. **결과 보고 후**: positive-dominant 구조로 stance_cfg 재설계
3. **Stage1 안정화 후**: 커리큘럼·도메인 랜덤화 도입

`TRAINING_HISTORY.md` §10 Option A(15차 회귀)는 권장하지 않는다 — 같은 local optimum에 다시 갇힐 뿐이다. **현재 구조를 유지하면서 1개 변수(termination)만 정정**하는 것이 최소 비용·최대 정보의 다음 step이다.

---

## 부록 A. 참고문헌

- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2021). *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*. CoRL 2021.
- Margolis, G., & Agrawal, P. (2023). *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*. CoRL 2022.
- Cheng, X., Shi, K., Agarwal, A., & Pathak, D. (2024). *Extreme Parkour with Legged Robots*. ICRA 2024.
- Zhuang, Z., Fu, Z., Wang, J., Atkeson, C., Schwertfeger, S., Finn, C., & Zhao, H. (2023). *Robot Parkour Learning*. CoRL 2023.
- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020). *Learning quadrupedal locomotion over challenging terrain*. Science Robotics.

## 부록 B. 진단용 텐서보드 요약 보드

학습 중 모니터링할 패널 권장 구성:

```
Row 1 (학습 신호):    ep_len_mean | mean_reward | entropy | action_std
Row 2 (보상 분해):    rew/alive | rew/upright | rew/gravity | rew/foot_slip
Row 3 (진단):         diag/per_step_net | diag/per_step_neg_only | diag/term_height | diag/term_tilt
Row 4 (PPO 내부):     value_loss | policy_loss | learning_rate | clip_fraction
```

`diag/per_step_net` < 0이 지속되면 즉시 학습을 중단하고 보상 scale을 재조정한다.
