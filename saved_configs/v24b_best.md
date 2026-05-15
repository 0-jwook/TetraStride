# v24b — Best Checkpoint Config

**날짜**: 2026-05-15  
**체크포인트**: `logs/rsl_rl/spot_micro_trot/2026-05-15_12-11-32/model_4999.pt`  
**전이 기저**: `logs/rsl_rl/spot_micro_trot/2026-05-14_18-56-54/model_4999.pt`

## 성능 지표

| 지표 | 값 |
|------|----|
| `actual_lin_vel_x` | 0.38 m/s |
| `heading_err_deg` | 0.76° |
| `lateral_drift_m` | 0.065 m |
| `rew/gait` | 17.4 / 20 |
| `rew/diagonal_symmetry` | -0.37 |
| `rew/air_time` | 0.103 |
| `rew/foot_height` | 0.079 |
| `term_ratio` | 0.0% |

## 시각적 평가

- 진동 감소 ✅
- 발 들기 동작 확인 ✅
- 완벽한 직선 보행 ✅
- 걸음걸이 개선 여지 있음 (다리를 더 시원시원하게 들어야 함)

---

## trot_cfg 전체 설정

```python
episode_length_s: float = 20.0
target_body_height: float = 0.17
action_scale: float = 0.35
# action_smoothing = 0.8 (전이학습 중 변경 금지)

cmd_lin_vel_x_range: tuple = (0.3, 0.7)
cmd_lin_vel_y_range: tuple = (0.0, 0.0)
cmd_ang_vel_z_range: tuple = (0.0, 0.0)
zero_command_prob: float = 0.1
gait_freq_hz: float = 1.5

# 1순위: Gait
rew_scale_gait: float = 5.0
rew_scale_air_time: float = 8.0
air_time_threshold: float = 0.12
rew_scale_swing_contact: float = -0.8
rew_scale_foot_height: float = 4.0
rew_scale_diagonal_symmetry: float = -1.0
rew_scale_air_time_var: float = 5.0

# 2순위: 방향
rew_scale_heading: float = 12.0
heading_sigma: float = 0.025
rew_scale_pos_drift: float = -8.0
rew_scale_yaw_tracking: float = 2.0
rew_scale_ang_vel_z: float = -2.0
rew_scale_lin_vel_xy: float = -2.0
rew_scale_heading_linear: float = -3.0
rew_scale_yaw_rate_error: float = -2.0

# 3순위: 속도
rew_scale_lin_vel: float = 6.0
rew_scale_ang_vel: float = 0.5
rew_scale_movement: float = 2.0
rew_scale_lin_vel_penalty: float = 0.0
rew_scale_alive: float = 0.5

# 자세 안정
rew_scale_body_height: float = -8.0
rew_scale_upright: float = 2.0
rew_scale_gravity: float = -5.0
rew_scale_ang_vel_xy: float = -0.5
rew_scale_lin_vel_z: float = -2.0

# 관절/토크 제약
rew_scale_joint_vel: float = -1e-4
rew_scale_torque: float = -1e-5
rew_scale_action_rate: float = -0.10
rew_scale_action_jerk: float = -0.03
rew_scale_dof_acc: float = -2e-6
rew_scale_contact_forces: float = -0.05
max_foot_contact_force: float = 30.0
rew_scale_termination: float = -5.0

# 자세 유지
target_foot_span: float = 0.10
rew_scale_joint_default: float = -8.0
rew_scale_foot_spread: float = -10.0
rew_scale_foot_slip: float = -1.5

# 무릎 보행 방지
non_foot_contact_threshold: float = 4.0
rew_scale_non_foot_contact: float = -5.0
rew_scale_stumble: float = -2.0
rew_scale_foot_stance: float = 2.0
rew_scale_knee_angle: float = -5.0
rew_scale_knee_height_stance: float = -10.0

# 기타
rew_scale_energy: float = 0.0
push_interval_s: float = 8.0
max_push_vel: float = 0.3
```

## PPO 설정

```python
num_steps_per_env = 24
max_iterations = 5000
save_interval = 200
experiment_name = "spot_micro_trot"
resume = True
load_run = "2026-05-14_18-56-54"
load_checkpoint = "model_4999.pt"
init_noise_std = 1.0
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
activation = "elu"
entropy_coef = 0.015
learning_rate = 1.0e-3
schedule = "adaptive"
gamma = 0.99
lam = 0.95
desired_kl = 0.01
```
