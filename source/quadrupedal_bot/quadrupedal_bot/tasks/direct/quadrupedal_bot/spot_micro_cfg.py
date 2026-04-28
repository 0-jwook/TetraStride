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
# Standing pose (body z ≈ 0.22 m):
#   leg = 0.5 rad (thigh slightly forward),  foot = -0.5 rad (calf roughly vertical)
#   Vertical reach: cos(0.5)*0.1075 + cos(0)*0.130 ≈ 0.094 + 0.130 = 0.224 m

SPOT_MICRO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/wodnr/Downloads/spot_micro.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        root_link_name="base_link",
        joint_drive=None,  # DCMotorCfg가 액추에이터를 담당하므로 URDF 드라이브 비활성화
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
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_shoulder": 0.0,
            ".*_leg": 0.5,
            ".*_foot": -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "shoulder_joints": DCMotorCfg(
            joint_names_expr=[".*_shoulder"],
            effort_limit=1.5,
            saturation_effort=1.5,
            velocity_limit=6.0,
            stiffness=15.0,
            damping=0.25,
        ),
        "leg_joints": DCMotorCfg(
            joint_names_expr=[".*_leg"],
            effort_limit=1.5,
            saturation_effort=1.5,
            velocity_limit=6.0,
            stiffness=15.0,
            damping=0.25,
        ),
        "foot_joints": DCMotorCfg(
            joint_names_expr=[".*_foot"],
            effort_limit=1.5,
            saturation_effort=1.5,
            velocity_limit=6.0,
            stiffness=15.0,
            damping=0.25,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration for the Spot Micro quadruped robot.

Robot: Spot Micro (hobby quadruped, 12 DOF)
URDF:  /home/wodnr/Downloads/spot_micro.urdf
       meshes at /home/wodnr/Desktop/urdf/stl/

Isaac Lab converts URDF → USD automatically on first run and caches to ~/.cache/isaaclab/.
If the URDF or meshes change, delete the cache and restart.

Actuator model: DCMotor (implicit PD).
Tune stiffness/damping if the robot oscillates or is too stiff at start.

Sim-to-real servo mapping: see scripts/real_robot_deploy.py
"""
