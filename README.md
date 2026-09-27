# Humanoid experiments

Small experiments for humanoid simulation and control with NVIDIA Isaac Sim / Isaac Lab.

## Unitree G1 keyboard teleoperation

This example runs the published pretrained RSL-RL locomotion policy for Unitree G1
and replaces Isaac Lab's sampled `base_velocity` command with keyboard input.

### Tested stack

- Ubuntu 24.04
- NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver 580 series
- Python 3.11
- Isaac Sim 5.1
- Isaac Lab v2.3.2
- RSL-RL

### Prerequisites

Set up Isaac Lab first and install the RSL-RL dependencies.

For example, if Isaac Lab lives at `~/lab/IsaacLab`:

```bash
cd ~/lab/IsaacLab
source .venv/bin/activate
./isaaclab.sh -i rsl_rl
```

The same Isaac Lab virtual environment can then be reused from this repository.

### Run

```bash
git clone https://github.com/remmaTech12/humanoid.git
cd humanoid

source ~/lab/IsaacLab/.venv/bin/activate
python g1_keyboard.py
```

The script downloads/uses the published pretrained checkpoint for
`Isaac-Velocity-Flat-G1-v0`.

### Controls

| Key | Command |
| --- | --- |
| Up / Numpad 8 | Forward |
| Down / Numpad 2 | Backward |
| Left / Right | Lateral motion |
| Z / X | Yaw left / right |
| L | Reset velocity command to zero |

The default command sensitivity can be changed from the command line:

```bash
python g1_keyboard.py --vx 0.6 --vy 0.3 --wz 0.8
```

Run `python g1_keyboard.py --help` for the remaining simulator options.

### How it works

The pretrained policy still produces the G1 joint actions. The only part changed by
this script is the locomotion command supplied to the policy:

```text
keyboard
   |
   v
Se2Keyboard
   |
   v
(vx, vy, wz)
   |
   v
base_velocity command
   |
   v
pretrained RSL-RL policy
   |
   v
G1 joint actions
```

This keeps the upstream Isaac Lab repository untouched while allowing experiments
to live in a separate project repository.
