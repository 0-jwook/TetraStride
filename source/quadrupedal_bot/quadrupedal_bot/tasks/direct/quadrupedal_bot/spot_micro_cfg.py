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
# Standing pose (body z ≈ 0.19~0.20 m, bent knee like real robot):
#   leg = 0.83 rad (47°), foot = -0.83 rad — real robot measurement Q2=-0.83, Q3=1.66
#   hip gravity torque ≈ 1.03 N·m — kp=20: gravity_sag=0.052 rad (vs kp=5: 0.206 rad)
#   equilibrium torque = 1.03 N·m < effort_limit=2.0 → NO saturation at standing pose

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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.22),
        joint_pos={
            ".*_shoulder": 0.0,
            ".*_leg": 0.83,
            ".*_foot": -0.83,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "shoulder_joints": DCMotorCfg(
            joint_names_expr=[".*_shoulder"],
            effort_limit=2.0,
            saturation_effort=2.0,
            velocity_limit=6.0,
            stiffness=15.0,   # 5→15: 어깨 중력 토크 작음 (abduction), 낮게 설정
            damping=0.5,
        ),
        "leg_joints": DCMotorCfg(
            joint_names_expr=[".*_leg"],
            effort_limit=2.0,
            saturation_effort=2.0,
            velocity_limit=6.0,
            stiffness=20.0,   # 5→20: gravity_sag 0.206→0.052 rad, crouch-to-survive 방지
            damping=0.6,
        ),
        "foot_joints": DCMotorCfg(
            joint_names_expr=[".*_foot"],
            effort_limit=2.0,
            saturation_effort=2.0,
            velocity_limit=6.0,
            stiffness=20.0,   # 5→20: knee joint 동일 강화
            damping=0.5,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration for the Spot Micro quadruped robot."""
