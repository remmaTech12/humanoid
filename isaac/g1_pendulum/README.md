# Unitree G1 — Passive Pendulum (Isaac Lab)

Phase 1 of the right-hand inverted-pendulum project: **physics only; no RL and no arm motion**.

- Real Isaac Lab \`G1_29DOF_CFG\` USD (Isaac Lab 2.3.2 / Isaac Sim 5.1)
- Every revolute/prismatic joint of the humanoid is converted to a **physical fixed joint**, and the pelvis is fixed to the world. The G1 is therefore immobile; there are no joint targets or PD controllers.
- A **1.0 m rigid cylinder**, pivoted at \`right_wrist_yaw_link\`, is connected by a **3-rotation-DOF USD spherical joint**. No cone limits, springs or drives.
- Cylinder mass **0.20 kg** plus central M5Stack ATOM approximation **0.02 kg**. Inertia includes the central mass; the blue cube is visual only, not a physical IMU yet.
- The rod has **no collision shapes**: G1, ground and other contact effects are excluded for Phase 1. The root is constrained only by the ball joint.
- Initial rod orientation is configurable by WORLD-frame X/Y tilt angles; gravity makes it fall.
- Angle/axis-direction data are recorded to CSV.

## Environment

Use the same environment as \`isaac/g1/\` in this repository:

- Ubuntu 24.04, NVIDIA GPU / drivers for Isaac Sim Kit (even if physics runs on CPU)
- Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2
- Isaac Lab's \`isaaclab_assets\` package and its G1 asset
- Network/cache access to the NVIDIA G1 USD on first use

Activate your existing Isaac Lab venv in the root of this repository:

~~~bash
source ~/lab/IsaacLab/.venv/bin/activate
python isaac/g1_pendulum/simulate.py --device cpu
~~~

Initial tilt default is **+5 degrees about WORLD Y**, so the rod points slightly toward WORLD +X. Camera, robot and pivot are fixed.

### Change only the initial rod orientation

~~~bash
# Tilt toward WORLD +X (rotation about +Y).
python isaac/g1_pendulum/simulate.py --tilt-y-deg 5 --tilt-x-deg 0 --device cpu

# Tilt toward WORLD -Y (rotation about +X).
python isaac/g1_pendulum/simulate.py --tilt-x-deg 5 --tilt-y-deg 0 --device cpu

# Three-dimensional initial lean; fail if the tilt never grows.
python isaac/g1_pendulum/simulate.py --tilt-x-deg 3 --tilt-y-deg 5 --assert-fall --device cpu

# Headless run and change the logging path.
python isaac/g1_pendulum/simulate.py --headless --device cpu --duration 4 --csv /tmp/g1_fall.csv
~~~

Arguments: \`--tilt-x-deg\`, \`--tilt-y-deg\`, \`--duration\`, \`--csv\`, \`--print-every\`, \`--assert-fall\`, and AppLauncher flags such as \`--headless\`.

**Do not use exactly zero tilt for this test:** an ideal inverted pendulum at precisely vertical with zero angular velocity is an unstable *equilibrium* and may remain stationary in exact arithmetic.

## Coordinate frames

| Frame | Definition |
| --- | --- |
| World | +Z up, +X/+Y form a right-handed horizontal plane |
| G1 | USD-published rest pose, pelvis fixed to World |
| Pivot | \`right_wrist_yaw_link\` local origin (placeholder for a physical palm adapter) |
| Rod | Local +Z points from pivot toward tip; origin at rod center |
| ATOM | Center of rod; blue cube is purely visual |

The two independent initial Euler tilt parameters are composed as \`R = R_y(tilt_y) R_x(tilt_x)\`. All angles are degrees. The rod tilt \`theta\` is the unsigned angle between the rod long axis and WORLD +Z, computed by \`acos(u_z)\` where \`u = R_world_rod [0,0,1]\`. The CSV also contains signed \`atan2(u_x,u_z)\` and \`atan2(u_y,u_z)\`.

## First-stage acceptance check

With a nonzero initial tilt and no actions, the tilt should initially grow due to gravity. Start with \`--tilt-y-deg 5 --duration 3 --assert-fall\` and verify the reported maximum tilt exceeds the initial tilt. Open \`isaac/g1_pendulum/logs/fall.csv\` for more details.

The program checks that the G1 USD contains movable joints, that a real right-wrist rigid body exists, and that the ball joint connects the right wrist and the rod.

### Implementation notes / limitations

- **G1 is in its authored USD rest pose** (not an IK-designed arm-extended palm-up pose). The spherical pivot is at the right-wrist link origin; the adapter geometry and exact palm TCP come later.
- All motor joints are replaced in a **layer override of the spawned stage**, not in the original USD. Other G1 demos are untouched.
- The robot is completely rigid only for this first test. Do not reuse the fixed-joint G1 in a future arm-control task without reverting that conversion.
- No IMU stream, collision detection, contact forces, angle limits, reward function or PPO are implemented.
- If an NVIDIA Isaac Lab update changes the G1 link/joint schema, verify and adapt the USD prim names in \`simulate.py\`.
- A visual run on a machine with Isaac Sim 5.1 / Lab 2.3.2 is still required; this code has not been runtime-tested in Isaac Sim here.

## Lightweight metric check

Run without Isaac Sim:

~~~bash
python -m unittest discover -s isaac/g1_pendulum/tests -v
~~~
