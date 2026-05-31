import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Joint structure (12 DOFs):
#   shoulder (abduction, X-axis): {front/rear}_{left/right}_shoulder  ±0.548 rad
#   leg (hip flex/ext, Y-axis):   {front/rear}_{left/right}_leg       [-2.666, 1.548] rad
#   foot (knee, Y-axis):          {front/rear}_{left/right}_foot      [-2.600, 0.100] rad
#
# Servo direction convention (from real robot):
#   hip sim+  →  servo+  for ALL hips  (direction=+1)
#   left  thigh/calf sim+  →  servo-  (direction=-1, sim+ = forward, servo+ = backward)
#   right thigh/calf sim+  →  servo+  (direction=+1, sim+ = forward, servo+ = forward)
#
# Standing pose (body z ≈ 0.17 m, proper reverse-knee):
#   leg = 0.83 rad (47°), foot = -1.55 rad — 역관절: calf 41° 앞으로, 발이 hip 직하방
#   foot=-0.83(구): calf 수직 → 발이 hip 뒤 → CoM 앞쏠림 → 마찰 의존 불안정 자세
#   foot=-1.55(신): body_height=0.17m, 발이 CoM 직하방 → 물리적으로 안정
#
# MG996R @6V (공식 TowerPro 스펙):
#   stall_torque = 11 kgf·cm = 1.08 N·m  → effort_limit=1.1
#   no_load_speed = 0.14 s/60° = 7.5 rad/s → velocity_limit=7.5
#   kp=30 + effort=1.08: 포화점 2.1° → kp=10으로 낮춰 6.3°까지 선형 제어 확보
#   overdamping: critical(kp=10, I≈0.001~0.005)=0.2~0.45 → damping=0.5으로 안전 오버댐핑

SPOT_MICRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/wodnr/Downloads/spot_micro_light.urdf",  # 2.5kg: hip 토크 24%, 포화 없음
        fix_base=False,
        merge_fixed_joints=True,
        root_link_name="base_link",
        joint_drive=None,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22),  # B-v13: 0.18→0.22 (높은 스폰으로 leg 완전 뻗기 유도)
        joint_pos={
            ".*_shoulder": 0.0,
            ".*_leg": 0.83,
            ".*_foot": -1.48,   # new-v12: -2.59→-1.48 복원 (PD 스프링 기준점=서기, action=0이 서기 보조)
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "shoulder_joints": DCMotorCfg(
            joint_names_expr=[".*_shoulder"],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=7.5,     # MG996R @6V: 7.5 rad/s (기존 6.0은 6rad/s 초과시 토크=0 버그)
            stiffness=20.0,
            damping=0.5,
        ),
        "leg_joints": DCMotorCfg(
            joint_names_expr=[".*_leg"],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=7.5,     # MG996R @6V: 7.5 rad/s
            stiffness=80.0,         # v29:30 → v30:80, 중력 처짐 0.007rad으로 감소
            damping=2.0,
        ),
        "foot_joints": DCMotorCfg(
            joint_names_expr=[".*_foot"],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=7.5,
            stiffness=80.0,         # v29:30 → v30:80, 역관절 주저앉음 방지 강화
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration for the Spot Micro quadruped robot."""
