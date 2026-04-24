"""Real robot deployment: Spot Micro RL policy를 실제 로봇에서 WASD로 제어.

Observation space (50-dim, 학습과 동일):
  [0:3]   base_lin_vel_b     (IMU + velocity estimator)
  [3:6]   base_ang_vel_b     (IMU gyroscope)
  [6:9]   projected_gravity  (IMU accelerometer normalized)
  [9:12]  commands           (vx, vy, wz) <- WASD로 제어
  [12:24] joint_pos_rel      (encoders - default_pos)
  [24:36] joint_vel          (encoders or finite diff)
  [36:48] last_actions       (이전 스텝 action)
  [48]    sin(gait_phase)    (1.5 Hz 클록)
  [49]    cos(gait_phase)    (1.5 Hz 클록)

WASD 제어 (Stage 3 학습 범위 내):
  W / S   : 전진(+0.7) / 후진(-0.4)   vx 범위: (-0.5, 1.0)
  A / D   : 좌이동(+0.3) / 우이동(-0.3)  vy 범위: (-0.3, 0.3)
  Q / E   : 좌회전(+0.5) / 우회전(-0.5) wz 범위: (-0.5, 0.5)
  Space/X : 정지   Ctrl+C: 종료

Servo mapping (from physical robot calibration):
  Joint order: FL_hip, FL_thigh, FL_calf,
               FR_hip, FR_thigh, FR_calf,
               RL_hip, RL_thigh, RL_calf,
               RR_hip, RR_thigh, RR_calf
  Direction: left thigh/calf -1 (sim+ → servo-), right/hips +1
"""

import math
import sys
import termios
import tty
import time
import threading
import numpy as np

# ---------------------------------------------------------------------------
# Servo mapping
# ---------------------------------------------------------------------------

SERVO_NEUTRAL_DEG = [
    90.0, 89.0, 91.0,    # FL hip, thigh, calf
   100.0, 76.0, 89.0,    # FR hip, thigh, calf
    90.0, 99.0, 91.0,    # RL hip, thigh, calf
    90.0, 91.0, 103.0,   # RR hip, thigh, calf
]

SERVO_DIRECTION = [
    +1, -1, -1,   # FL
    +1, +1, +1,   # FR
    +1, -1, -1,   # RL
    +1, +1, +1,   # RR
]

SERVO_STANDING_DEG = [
    n + d * (0.0 if i % 3 == 0 else 0.5 if i % 3 == 1 else -0.5) * (180 / math.pi)
    for i, (n, d) in enumerate(zip(SERVO_NEUTRAL_DEG, SERVO_DIRECTION))
]

DEFAULT_JOINT_POS_RAD = np.array(
    [0.0, 0.5, -0.5] * 4, dtype=np.float32
)

KEY_CMD_MAP = {
    'w': ( 0.7,  0.0,  0.0),
    's': (-0.4,  0.0,  0.0),
    'a': ( 0.0,  0.3,  0.0),
    'd': ( 0.0, -0.3,  0.0),
    'q': ( 0.0,  0.0,  0.5),
    'e': ( 0.0,  0.0, -0.5),
    ' ': ( 0.0,  0.0,  0.0),
    'x': ( 0.0,  0.0,  0.0),
}


# ---------------------------------------------------------------------------
# Servo utilities
# ---------------------------------------------------------------------------

def sim_joints_to_servo(sim_rad: np.ndarray) -> np.ndarray:
    n = np.array(SERVO_NEUTRAL_DEG, dtype=np.float32)
    d = np.array(SERVO_DIRECTION, dtype=np.float32)
    return np.clip(n + d * np.degrees(sim_rad), 0.0, 180.0)


def servo_to_sim_joints(servo_deg: np.ndarray) -> np.ndarray:
    n = np.array(SERVO_NEUTRAL_DEG, dtype=np.float32)
    d = np.array(SERVO_DIRECTION, dtype=np.float32)
    return np.radians((servo_deg - n) / d)


# ---------------------------------------------------------------------------
# Non-blocking WASD keyboard reader
# ---------------------------------------------------------------------------

class KeyboardController:
    def __init__(self):
        self.cmd = [0.0, 0.0, 0.0]
        self._running = False
        self._fd = None
        self._old_settings = None

    def start(self):
        self._running = True
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def _read_loop(self):
        tty.setraw(self._fd)
        try:
            while self._running:
                ch = sys.stdin.read(1).lower()
                if ch in KEY_CMD_MAP:
                    self.cmd = list(KEY_CMD_MAP[ch])
                    vx, vy, wz = self.cmd
                    print(f"\r[CMD] vx={vx:+.1f} vy={vy:+.1f} wz={wz:+.1f}   ", end='', flush=True)
                elif ch == '\x03':  # Ctrl+C
                    self._running = False
                    break
        finally:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


# ---------------------------------------------------------------------------
# Hardware interface placeholder
# ---------------------------------------------------------------------------

class RealRobotInterface:
    """실제 서보 하드웨어로 교체 필요 (PCA9685, Dynamixel 등)."""

    def connect(self):
        print("[RealRobot] Connected (placeholder).")

    def send_servo_commands(self, servo_deg: np.ndarray):
        print(f"\r[Servo] {np.round(servo_deg, 1).tolist()}", end='', flush=True)

    def read_joint_angles(self):
        return None

    def close(self):
        print("\n[RealRobot] Disconnected.")


# ---------------------------------------------------------------------------
# Main policy runner
# ---------------------------------------------------------------------------

def run_policy_on_real_robot(
    policy_path: str,
    num_steps: int = 10000,
    dt: float = 0.02,        # decimation(4) / sim_freq(200Hz) = 0.02s
    action_scale: float = 0.5,
):
    import torch

    print("=" * 55)
    print("  Spot Micro WASD 제어")
    print("  W/S:전진/후진  A/D:좌우이동  Q/E:좌우회전")
    print("  Space/X:정지   Ctrl+C:종료")
    print("=" * 55)

    policy = torch.jit.load(policy_path)
    policy.eval()

    robot = RealRobotInterface()
    robot.connect()
    robot.send_servo_commands(np.array(SERVO_STANDING_DEG, dtype=np.float32))
    time.sleep(1.5)
    print("\n[INFO] 서기 자세 완료. 조종 시작.")

    keyboard = KeyboardController()
    keyboard.start()

    obs = torch.zeros(1, 50)
    last_action = np.zeros(12, dtype=np.float32)
    gait_phase = 0.0

    try:
        for _ in range(num_steps):
            t0 = time.time()

            # gait phase 업데이트 (학습과 동일: 1.5 Hz)
            gait_phase = (gait_phase + dt * 2.0 * math.pi * 1.5) % (2.0 * math.pi)

            # Observation 구성
            # IMU/엔코더 없을 경우 obs[0:12], obs[12:36]은 0으로 유지
            vx, vy, wz = keyboard.cmd
            obs[0, 9]  = vx
            obs[0, 10] = vy
            obs[0, 11] = wz
            obs[0, 36:48] = torch.from_numpy(last_action)
            obs[0, 48] = math.sin(gait_phase)
            obs[0, 49] = math.cos(gait_phase)

            with torch.no_grad():
                action = policy(obs).squeeze(0).numpy()

            sim_angles = DEFAULT_JOINT_POS_RAD + action * action_scale
            robot.send_servo_commands(sim_joints_to_servo(sim_angles))
            last_action = action.copy()

            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자 중단.")
    finally:
        keyboard.stop()
        robot.send_servo_commands(np.array(SERVO_STANDING_DEG, dtype=np.float32))
        time.sleep(0.5)
        robot.close()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _verify_mapping():
    REFERENCE_SIM = np.array(
        [0.0, 1.548, -1.548] * 4, dtype=np.float32
    )
    REFERENCE_SERVO = [90.0, 0.0, 180.0, 100.0, 165.0, 0.0,
                       90.0, 10.0, 180.0, 90.0, 180.0, 14.2]
    computed = sim_joints_to_servo(REFERENCE_SIM)
    names = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
             "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]
    print("=== Servo mapping verification ===")
    print(f"{'Joint':<12} {'Expected':>10} {'Computed':>10} {'Error':>8}")
    print("-" * 44)
    for name, exp, comp in zip(names, REFERENCE_SERVO, computed):
        print(f"{name:<12} {exp:>10.1f} {comp:>10.1f} {comp-exp:>+8.2f}")
    print("\n=== Standing servo angles ===")
    for name, angle in zip(names, SERVO_STANDING_DEG):
        print(f"{name:<12} {angle:>10.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    if args.verify or args.policy is None:
        _verify_mapping()
    else:
        run_policy_on_real_robot(args.policy, num_steps=args.steps, dt=args.dt)
