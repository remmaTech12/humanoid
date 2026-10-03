# Unitree G1

Experiments for Unitree G1 simulation and control with NVIDIA Isaac Sim / Isaac Lab.

## Keyboard teleoperation

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
python isaac/g1/keyboard.py
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
python isaac/g1/keyboard.py --vx 0.6 --vy 0.3 --wz 0.8
```

Run `python isaac/g1/keyboard.py --help` for the remaining simulator options.

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


## Policy fine-tuning

`train.py` fine-tunes the published pretrained G1 Flat RSL-RL policy
instead of starting from random weights. Velocity command ranges are configurable,
so the script is not limited to backward walking.

It keeps the Isaac Lab task fixed to:

```text
Isaac-Velocity-Flat-G1-v0
```

The velocity command ranges, PPO learning rate, initial checkpoint, and output
policy name can be selected from the command line.

### Quick smoke test

From this repository, with the Isaac Lab virtual environment activated:

```bash
source ~/lab/IsaacLab/.venv/bin/activate

python isaac/g1/train.py \
  --checkpoint ~/lab/IsaacLab/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-G1-v0/checkpoint.pt \
  --policy-name backward \
  --vx -0.4 -0.1 \
  --vy 0.0 0.0 \
  --wz 0.0 0.0 \
  --learning-rate 1e-3 \
  --num-envs 64 \
  --max-iterations 50
```

The checkpoint shown above is the same published pretrained
`Isaac-Velocity-Flat-G1-v0` policy used by `keyboard.py` by default.

The initial policy weights are loaded from the checkpoint, but the optimizer is
started fresh so that `--learning-rate` is applied exactly as requested.

### Output

A run is stored under:

```text
isaac/g1/policies/<policy-name>/<timestamp>/
```

For example:

```text
isaac/g1/policies/backward/2026-10-03_17-30-00/
├── model_0.pt
├── model_49.pt
├── params/
│   ├── agent.yaml
│   └── env.yaml
└── events.out.tfevents...
```

The `policies` directory is ignored by Git because checkpoints can be large.
