"""action=0 고정으로 PD 컨트롤러만으로 로봇이 어디로 수렴하는지 측정.

목적:
  1. action=0에서 default joint pos를 유지하는가?
  2. 유지한다면 20초 (2400 steps) 동안 안정적인가?
  3. 어떻게 무너지는가? (높이, 자세, 관절 변화)

이게 RL 없이 봤어야 했던 베이스라인 정보입니다.
"""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Template-Quadrupedal-Bot-Stance-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
import quadrupedal_bot.tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    uw = env.unwrapped
    leg_ids, _ = uw.robot.find_joints(".*_leg")
    knee_ids, _ = uw.robot.find_joints(".*_foot")

    action_dim = uw.action_space.shape[-1] if uw.action_space.shape[-1] != args_cli.num_envs else uw.num_actions
    action = torch.zeros(args_cli.num_envs, action_dim, device=uw.device)
    print(f"[DEBUG] action shape: {action.shape}")

    print()
    print("=" * 105)
    print("EQUILIBRIUM TEST - action=0 (PD controller only, no policy)")
    print("Target: hip=0.83, knee=-1.48, height=0.177m, alive 20s (2400 steps)")
    print("=" * 105)
    print(f"{'step':>5} | {'time':>5} | {'h_mean':>7} | {'h_min':>7} | "
          f"{'hip_FL':>7} | {'hip_FR':>7} | {'hip_RL':>7} | {'hip_RR':>7} | "
          f"{'knee_FL':>8} | {'knee_FR':>8} | {'alive':>5}")
    print("-" * 105)

    alive_mask = torch.ones(args_cli.num_envs, dtype=torch.bool, device=uw.device)

    for step in range(2400):
        obs, rew, terminated, truncated, info = env.step(action)
        alive_mask = alive_mask & ~terminated

        # 처음 30 step은 매 step, 그 이후는 50step마다
        log_this = (step < 30) or (step % 50 == 0) or (step == 2399) or (alive_mask.sum().item() == 0)
        if log_this:
            height = uw.robot.data.root_pos_w[:, 2]
            jpos = uw.robot.data.joint_pos

            if alive_mask.any():
                live_h = height[alive_mask]
                live_jpos = jpos[alive_mask]
                h_mean = live_h.mean().item()
                h_min = live_h.min().item()
                hip_FL = live_jpos[:, leg_ids[0]].mean().item()
                hip_FR = live_jpos[:, leg_ids[1]].mean().item()
                hip_RL = live_jpos[:, leg_ids[2]].mean().item()
                hip_RR = live_jpos[:, leg_ids[3]].mean().item()
                knee_FL = live_jpos[:, knee_ids[0]].mean().item()
                knee_FR = live_jpos[:, knee_ids[1]].mean().item()
            else:
                h_mean = h_min = hip_FL = hip_FR = hip_RL = hip_RR = knee_FL = knee_FR = float('nan')

            alive_count = alive_mask.sum().item()
            print(f"{step:5d} | {step*0.00833:4.2f}s | {h_mean:7.4f} | {h_min:7.4f} | "
                  f"{hip_FL:7.4f} | {hip_FR:7.4f} | {hip_RL:7.4f} | {hip_RR:7.4f} | "
                  f"{knee_FL:8.4f} | {knee_FR:8.4f} | {alive_count:>2}/{args_cli.num_envs}")

            if alive_count == 0:
                print(f"\n[ALL DEAD at step {step}, time {step*0.00833:.2f}s]")
                break

    print()
    print("=" * 105)
    final_alive = alive_mask.sum().item()
    print(f"RESULT: {final_alive}/{args_cli.num_envs} robots survived 20 seconds")
    print("=" * 105)
    if final_alive > 0:
        print(f"  Final height: {h_mean:.4f}m (target 0.177m)")
        print(f"  Final hip:    FL={hip_FL:.3f} FR={hip_FR:.3f} RL={hip_RL:.3f} RR={hip_RR:.3f}  (target 0.83)")
        print(f"  Final knee:   FL={knee_FL:.3f} FR={knee_FR:.3f}  (target -1.48)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
