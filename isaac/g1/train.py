"""Fine-tune the Unitree G1 flat-ground locomotion policy.

This script intentionally reuses Isaac Lab's G1 Flat task and RSL-RL PPO setup.
The command ranges, learning rate, initial checkpoint, and output directory are
customizable from this repository.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

TASK_NAME = "Isaac-Velocity-Flat-G1-v0"
DEFAULT_POLICY_NAME = "backward"


def _range_pair(values: list[float], name: str) -> tuple[float, float]:
    """Validate and convert a two-value CLI range."""
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = values
    if low > high:
        raise ValueError(f"{name} minimum must be <= maximum: {values}")
    return float(low), float(high)


parser = argparse.ArgumentParser(
    description="Fine-tune the pretrained Unitree G1 Flat RSL-RL policy."
)
parser.add_argument(
    "--policy-name",
    type=str,
    default=DEFAULT_POLICY_NAME,
    help="Directory name under isaac/g1/policies/ used to store this policy.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Initial RSL-RL checkpoint. If omitted, the published pretrained "
        "Isaac-Velocity-Flat-G1-v0 checkpoint is used."
    ),
)
parser.add_argument(
    "--vx",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=[-0.4, -0.1],
    help="Command range for forward/backward velocity in m/s.",
)
parser.add_argument(
    "--vy",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=[0.0, 0.0],
    help="Command range for lateral velocity in m/s.",
)
parser.add_argument(
    "--wz",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=[0.0, 0.0],
    help="Command range for yaw rate in rad/s.",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=1.0e-3,
    help="PPO learning rate.",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=64,
    help="Number of parallel Isaac Lab environments.",
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=50,
    help="Number of PPO learning iterations.",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=50,
    help="Checkpoint save interval in learning iterations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def _resolve_checkpoint() -> str:
    """Resolve either a user checkpoint or Isaac Lab's published pretrained policy."""
    if args_cli.checkpoint is not None:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return str(checkpoint)

    checkpoint = get_published_pretrained_checkpoint("rsl_rl", TASK_NAME)
    if not checkpoint:
        raise RuntimeError(
            f"No published pretrained RSL-RL checkpoint is available for {TASK_NAME}."
        )
    return checkpoint


def _create_output_dir() -> Path:
    """Create policies/<policy-name>/<timestamp> inside this repository."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args_cli.policy_name):
        raise ValueError(
            "--policy-name may contain only letters, numbers, '.', '_' and '-'."
        )

    policy_root = Path(__file__).resolve().parent / "policies" / args_cli.policy_name
    run_dir = policy_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    """Fine-tune the published G1 Flat policy with custom velocity commands."""
    vx = _range_pair(args_cli.vx, "--vx")
    vy = _range_pair(args_cli.vy, "--vy")
    wz = _range_pair(args_cli.wz, "--wz")

    if args_cli.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args_cli.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    if args_cli.save_interval <= 0:
        raise ValueError("--save-interval must be positive.")

    device = args_cli.device if args_cli.device is not None else "cuda:0"

    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=device,
        num_envs=args_cli.num_envs,
    )
    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(
        TASK_NAME,
        "rsl_rl_cfg_entry_point",
    )

    # Keep the upstream G1 Flat task, but change only the commanded motion.
    env_cfg.commands.base_velocity.ranges.lin_vel_x = vx
    env_cfg.commands.base_velocity.ranges.lin_vel_y = vy
    env_cfg.commands.base_velocity.ranges.ang_vel_z = wz
    # G1 Flat normally uses heading targets to generate yaw commands. Disable
    # that mode so the explicit --wz range above is actually used.
    env_cfg.commands.base_velocity.heading_command = False

    # Fine-tuning parameters.
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    agent_cfg.device = device
    env_cfg.sim.device = device
    env_cfg.seed = agent_cfg.seed

    checkpoint = _resolve_checkpoint()
    output_dir = _create_output_dir()

    env_cfg.log_dir = str(output_dir)

    print(f"[INFO] Task             : {TASK_NAME}")
    print(f"[INFO] Initial checkpoint: {checkpoint}")
    print(f"[INFO] Output directory  : {output_dir}")
    print(f"[INFO] vx range          : {vx}")
    print(f"[INFO] vy range          : {vy}")
    print(f"[INFO] wz range          : {wz}")
    print(f"[INFO] Learning rate     : {args_cli.learning_rate}")
    print(f"[INFO] Environments      : {args_cli.num_envs}")
    print(f"[INFO] Max iterations    : {args_cli.max_iterations}")

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            f"Expected OnPolicyRunner for {TASK_NAME}, got {agent_cfg.class_name}."
        )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(output_dir),
        device=agent_cfg.device,
    )

    # Fine-tune model weights from the pretrained checkpoint, but start with a
    # fresh optimizer so the CLI learning rate is guaranteed to take effect.
    runner.load(checkpoint, load_optimizer=False)
    runner.current_learning_iteration = 0

    dump_yaml(str(output_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(output_dir / "params" / "agent.yaml"), agent_cfg)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
