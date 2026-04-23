"""Real robot deployment: converts trained RL policy output to servo commands.

Servo hardware convention (from physical robot):
  - Range: 0 ~ 180 degrees per servo
  - Positive direction effects:
      FL/RL hip   → shoulder lifts (abduction)
      FR/RR hip   → goes inward (adduction)
      FL/RL thigh → extends backward
      FR/RR thigh → extends forward
      FL/RL calf  → extends backward
      FR/RR calf  → extends forward

Extended-forward pose reference (all legs stretched forward, A: prefix):
  A: 90, 0, 180, 100, 165, 0, 90, 10, 180, 90, 180, 14.2
  Order: FL_hip, FL_thigh, FL_calf,
         FR_hip, FR_thigh, FR_calf,
         RL_hip, RL_thigh, RL_calf,
         RR_hip, RR_thigh, RR_calf

Joint order in Isaac Lab simulation (URDF traversal):
  0: front_left_shoulder   (FL hip)
  1: front_left_leg        (FL thigh)
  2: front_left_foot       (FL calf)
  3: front_right_shoulder  (FR hip)
  4: front_right_leg       (FR thigh)
  5: front_right_foot      (FR calf)
  6: rear_left_shoulder    (RL hip)
  7: rear_left_leg         (RL thigh)
  8: rear_left_foot        (RL calf)
  9: rear_right_shoulder   (RR hip)
  10: rear_right_leg       (RR thigh)
  11: rear_right_foot      (RR calf)
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Servo mapping parameters
# Derived from extended-forward reference pose assuming:
#   sim_hip_at_ref   ≈ +1.548 rad  (URDF max forward)
#   sim_calf_at_ref  ≈ -1.548 rad  (calf vertical → thigh angle - π/2)
# Formula: servo = neutral + direction × sim_rad × (180/π)
# ---------------------------------------------------------------------------

# Servo neutral angles (degrees) when sim joint angle = 0 rad
# Derived by back-calculating from the reference pose values
SERVO_NEUTRAL_DEG = [
    90.0,   # 0  FL hip   (ref: 90  = 90  + 1*(0)*57.3)
    89.0,   # 1  FL thigh (ref:  0  = 89  + (-1)*(1.548)*57.3 → 89-88.7=0.3≈0 ✓)
    91.0,   # 2  FL calf  (ref: 180 = 91  + (-1)*(-1.548)*57.3 → 91+88.7=179.7≈180 ✓)
   100.0,   # 3  FR hip   (ref: 100 = 100 + 1*(0)*57.3)
    76.0,   # 4  FR thigh (ref: 165 = 76  + (+1)*(1.548)*57.3 → 76+88.7=164.7≈165 ✓)
    89.0,   # 5  FR calf  (ref:   0 = 89  + (+1)*(-1.548)*57.3 → 89-88.7=0.3≈0 ✓)
    90.0,   # 6  RL hip
    99.0,   # 7  RL thigh (ref:  10 = 99  + (-1)*(1.548)*57.3 → 99-88.7=10.3≈10 ✓)
    91.0,   # 8  RL calf  (ref: 180 = 91  + (-1)*(-1.548)*57.3 ✓)
    90.0,   # 9  RR hip
    91.0,   # 10 RR thigh (ref: 180 = 91  + (+1)*(1.548)*57.3 → 91+88.7=179.7≈180 ✓)
   103.0,   # 11 RR calf  (ref:14.2 = 103 + (+1)*(-1.548)*57.3 → 103-88.7=14.3≈14.2 ✓)
]

# Direction: +1 means sim+ → servo increases, -1 means sim+ → servo decreases
# Left  thigh/calf: sim+ (forward) → servo- (they go backward) → direction -1
# Right thigh/calf: sim+ (forward) → servo+ (they go forward)  → direction +1
# All hips: direction +1 (sim+ maps to the physical positive direction for each side)
SERVO_DIRECTION = [
    +1,  # 0  FL hip   (sim+ = abduction/lifts  = servo+)
    -1,  # 1  FL thigh (sim+ = forward → servo backward)
    -1,  # 2  FL calf  (sim+ = forward → servo backward)
    +1,  # 3  FR hip   (sim+ = adduction/inward = servo+)
    +1,  # 4  FR thigh (sim+ = forward → servo forward)
    +1,  # 5  FR calf  (sim+ = forward → servo forward)
    +1,  # 6  RL hip
    -1,  # 7  RL thigh
    -1,  # 8  RL calf
    +1,  # 9  RR hip
    +1,  # 10 RR thigh
    +1,  # 11 RR calf
]

# Expected standing-pose servo angles (for reference / sanity check)
# Derived from sim standing pose: hip=0.5 rad, calf=-0.5 rad, shoulder=0 rad
SERVO_STANDING_DEG = [
    neutral + direction * 0.0 * (180 / math.pi)   # shoulder: sim=0
    if i in (0, 3, 6, 9)
    else neutral + direction * 0.5 * (180 / math.pi)   # thigh: sim=0.5
    if i in (1, 4, 7, 10)
    else neutral + direction * (-0.5) * (180 / math.pi)  # calf: sim=-0.5
    for i, (neutral, direction) in enumerate(zip(SERVO_NEUTRAL_DEG, SERVO_DIRECTION))
]


def sim_joints_to_servo(sim_joint_angles_rad: np.ndarray) -> np.ndarray:
    """Convert simulation joint angles (radians) to servo angles (degrees).

    Args:
        sim_joint_angles_rad: Array of shape (12,) with joint angles in radians.
            Joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
                         RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf

    Returns:
        servo_angles_deg: Array of shape (12,) with servo angles in degrees [0, 180].
    """
    neutral = np.array(SERVO_NEUTRAL_DEG, dtype=np.float32)
    direction = np.array(SERVO_DIRECTION, dtype=np.float32)
    servo_deg = neutral + direction * np.degrees(sim_joint_angles_rad)
    # clamp to physical servo range
    return np.clip(servo_deg, 0.0, 180.0)


def servo_to_sim_joints(servo_angles_deg: np.ndarray) -> np.ndarray:
    """Convert servo angles (degrees) back to simulation joint angles (radians).

    Useful for reading back the robot's actual joint state.
    """
    neutral = np.array(SERVO_NEUTRAL_DEG, dtype=np.float32)
    direction = np.array(SERVO_DIRECTION, dtype=np.float32)
    sim_rad = np.radians((servo_angles_deg - neutral) / direction)
    return sim_rad


# ---------------------------------------------------------------------------
# Sanity check: verify extended-forward reference pose round-trips correctly
# ---------------------------------------------------------------------------

REFERENCE_EXTENDED_FWD_SIM_RAD = [
    0.0,        # FL hip: neutral
    1.548,      # FL thigh: max forward
    -1.548,     # FL calf: vertical (thigh_angle - π/2 ≈ -0.023, approx -1.548 for vert.)
    0.0,        # FR hip
    1.548,      # FR thigh
    -1.548,     # FR calf
    0.0, 1.548, -1.548,
    0.0, 1.548, -1.548,
]

REFERENCE_EXTENDED_FWD_SERVO_DEG = [90.0, 0.0, 180.0, 100.0, 165.0, 0.0,
                                     90.0, 10.0, 180.0, 90.0, 180.0, 14.2]


def _verify_mapping():
    sim = np.array(REFERENCE_EXTENDED_FWD_SIM_RAD, dtype=np.float32)
    expected = np.array(REFERENCE_EXTENDED_FWD_SERVO_DEG, dtype=np.float32)
    computed = sim_joints_to_servo(sim)
    names = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    print("=== Mapping verification (extended-forward pose) ===")
    print(f"{'Joint':<12} {'Expected':>10} {'Computed':>10} {'Error':>8}")
    print("-" * 44)
    for name, exp, comp in zip(names, expected, computed):
        print(f"{name:<12} {exp:>10.1f} {comp:>10.1f} {comp-exp:>+8.2f}")

    print("\n=== Standing pose servo angles ===")
    print(f"{'Joint':<12} {'Servo(°)':>10}")
    print("-" * 24)
    for name, angle in zip(names, SERVO_STANDING_DEG):
        print(f"{name:<12} {angle:>10.1f}")


# ---------------------------------------------------------------------------
# Hardware interface placeholder
# Replace the body of send_servo_commands() with your actual hardware call.
# ---------------------------------------------------------------------------

class RealRobotInterface:
    """Placeholder interface for the real Spot Micro hardware.

    Replace the methods below with your actual servo controller code
    (e.g., PCA9685 via I²C, Dynamixel SDK, etc.).
    """

    def connect(self):
        print("[RealRobot] Connected (placeholder).")

    def send_servo_commands(self, servo_angles_deg: np.ndarray):
        """Send 12 servo angles (degrees) to the robot.

        Args:
            servo_angles_deg: Array of 12 angles, order:
                FL_hip, FL_thigh, FL_calf,
                FR_hip, FR_thigh, FR_calf,
                RL_hip, RL_thigh, RL_calf,
                RR_hip, RR_thigh, RR_calf
        """
        # --- replace this with your hardware code ---
        print(f"[RealRobot] Servo cmd: {np.round(servo_angles_deg, 1).tolist()}")

    def read_joint_angles(self) -> np.ndarray | None:
        """Read current servo angles from the robot (if encoders are available).

        Returns None if the hardware does not support feedback.
        """
        return None

    def close(self):
        print("[RealRobot] Disconnected.")


# ---------------------------------------------------------------------------
# Policy runner (standalone, no Isaac Sim needed)
# ---------------------------------------------------------------------------

def run_policy_on_real_robot(policy_path: str, num_steps: int = 1000, dt: float = 0.02):
    """Load a trained JIT policy and run it on the real robot.

    Args:
        policy_path: Path to the exported policy.pt (JIT script).
        num_steps:   Number of control steps to execute.
        dt:          Control loop period in seconds (should match sim: decimation/200Hz = 0.02s).
    """
    import time
    import torch

    policy = torch.jit.load(policy_path)
    policy.eval()

    robot = RealRobotInterface()
    robot.connect()

    # Standing servo targets (initial safe position)
    standing_servo = np.array(SERVO_STANDING_DEG, dtype=np.float32)
    robot.send_servo_commands(standing_servo)
    time.sleep(1.0)

    # Command: forward at 0.5 m/s, no lateral, no yaw
    cmd_lin_x = 0.5
    cmd_lin_y = 0.0
    cmd_ang_z = 0.0

    # Observation state (must match env observation order)
    # In a real deployment you need sensors (IMU + encoders) to fill these.
    # This is a simplified example that zeros out unavailable measurements.
    obs = torch.zeros(1, 48)

    default_sim_joints = np.radians([
        0.0, 28.6, -28.6,   # FL: shoulder=0, hip=0.5rad=28.6°, calf=-0.5rad
        0.0, 28.6, -28.6,   # FR
        0.0, 28.6, -28.6,   # RL
        0.0, 28.6, -28.6,   # RR
    ])

    last_servo = standing_servo.copy()

    try:
        for step in range(num_steps):
            t0 = time.time()

            # --- Build observation (48-dim) ---
            # In real deployment, fill from IMU and encoders:
            # obs[0, 0:3]  = base_lin_vel_body_frame  (IMU + velocity estimator)
            # obs[0, 3:6]  = base_ang_vel_body_frame   (IMU gyroscope)
            # obs[0, 6:9]  = projected_gravity_body    (IMU accelerometer normalized)
            # obs[0, 9:12] = velocity commands
            obs[0, 9]  = cmd_lin_x
            obs[0, 10] = cmd_lin_y
            obs[0, 11] = cmd_ang_z
            # obs[0, 12:24] = joint_pos - default_joint_pos  (encoders)
            # obs[0, 24:36] = joint_vel                       (encoders or finite diff)
            # obs[0, 36:48] = last_actions                    (stored from prev step)

            # --- Policy inference ---
            with torch.no_grad():
                action = policy(obs).squeeze(0).numpy()  # shape (12,)

            # --- Convert action to joint angles ---
            # action is a position offset (rad) from the default joint position
            action_scale = 0.5  # must match env cfg
            sim_joint_angles = default_sim_joints + action * action_scale

            # --- Convert to servo degrees ---
            servo_angles = sim_joints_to_servo(sim_joint_angles)
            robot.send_servo_commands(servo_angles)
            last_servo = servo_angles.copy()

            # --- Timing ---
            elapsed = time.time() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[RealRobot] Interrupted by user.")
    finally:
        # Return to standing before disconnecting
        robot.send_servo_commands(standing_servo)
        time.sleep(0.5)
        robot.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify servo mapping and exit.")
    parser.add_argument("--policy", type=str, default=None, help="Path to exported policy.pt")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    if args.verify or args.policy is None:
        _verify_mapping()
    else:
        run_policy_on_real_robot(args.policy, num_steps=args.steps)
