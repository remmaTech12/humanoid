"""Keyboard teleoperation for the Unitree G1 locomotion policy in Isaac Lab.

This script keeps the pretrained RSL-RL locomotion policy and replaces the
environment's sampled base-velocity command with keyboard input from Se2Keyboard.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate Unitree G1 with the keyboard.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-G1-v0",
    help="Isaac Lab task name.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to an RSL-RL checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--vx", type=float, default=0.8, help="Forward/backward velocity sensitivity.")
parser.add_argument("--vy", type=float, default=0.4, help="Lateral velocity sensitivity.")
parser.add_argument("--wz", type=float, default=1.0, help="Yaw-rate sensitivity.")
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Sleep to keep simulation near real time.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def main():
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(
        args_cli.task, "rsl_rl_cfg_entry_point"
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = device

    if args_cli.checkpoint is not None:
        checkpoint = args_cli.checkpoint
    else:
        checkpoint = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not checkpoint:
            raise RuntimeError(
                f"No published pretrained RSL-RL checkpoint is available for {train_task_name}."
            )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading pretrained checkpoint: {checkpoint}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.vx,
            v_y_sensitivity=args_cli.vy,
            omega_z_sensitivity=args_cli.wz,
            sim_device=str(env.unwrapped.device),
        )
    )
    print(keyboard)
    print("\n[INFO] Keyboard command overrides the environment base_velocity command.")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Replace the environment-generated command with keyboard input.
            manual_command = keyboard.advance()
            velocity_command = env.unwrapped.command_manager.get_command("base_velocity")
            velocity_command[:] = manual_command

            # Recompute observations so the policy sees the manual command immediately.
            obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
