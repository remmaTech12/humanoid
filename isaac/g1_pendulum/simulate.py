"""Passive 3D pendulum mounted to the right wrist of a physically locked Unitree G1.

Isaac Lab 2.3.2 / Isaac Sim 5.1. No policy, motors, pendulum angle limits,
self-contact, or ground contact for the pendulum in this first stage.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 right-wrist passive spherical pendulum")
parser.add_argument("--tilt-x-deg", type=float, default=0.0, help="Initial rod rotation about WORLD +X (degrees).")
parser.add_argument("--tilt-y-deg", type=float, default=5.0, help="Initial rod rotation about WORLD +Y (degrees).")
parser.add_argument("--duration", type=float, default=3.0, help="Simulation time in seconds.")
parser.add_argument("--print-every", type=int, default=24, help="Print every N physics steps.")
parser.add_argument(
    "--csv", type=Path, default=Path(__file__).resolve().parent / "logs" / "fall.csv",
    help="Trajectory CSV path.",
)
parser.add_argument(
    "--assert-fall", action="store_true",
    help="Exit with an error if tilt never increases by at least 2 degrees.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Validate without launching the heavy application.
if not all(math.isfinite(x) for x in (args.tilt_x_deg, args.tilt_y_deg, args.duration)):
    parser.error("Angles and duration must be finite.")
if abs(args.tilt_x_deg) >= 85 or abs(args.tilt_y_deg) >= 85:
    parser.error("Initial tilt magnitude must be less than 85 degrees on each axis.")
if args.duration <= 0:
    parser.error("--duration must be positive.")
if args.print_every < 1:
    parser.error("--print-every must be at least 1.")
# With use_fabric=False the USD stage is updated, and CPU PhysX can render it.
# This is one pendulum, so massive GPU-parallel simulation is unnecessary.
if args.device not in (None, "cpu"):
    parser.error("This stage uses CPU PhysX for USD pose logging. Pass --device cpu.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab_assets import G1_29DOF_CFG
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from metrics import rod_tilt_deg

ROBOT_PATH = "/World/G1"
ROD_PATH = "/World/Pendulum/Rod"
JOINT_PATH = "/World/Pendulum/SphericalJoint"
ROD_LENGTH_M = 1.0
ROD_RADIUS_M = 0.008
ROD_MASS_KG = 0.20
ATOM_MASS_KG = 0.02  # Placeholder: replace with the actual M5Stack ATOM + mount mass.
ATOM_SIDE_M = 0.03
# Offset is measured in the local wrist-link frame, not world.
PIVOT_IN_WRIST = (0.0, 0.0, 0.0)


def _lock_robot_joints(stage: Usd.Stage) -> int:
    """Replace every revolute/prismatic G1 joint by a physical fixed joint.

    Keeps the referenced USD's initial geometry/pose and joint local frames.
    This is a *true* physical lock; no position-drive gains are relied upon.
    """
    count = 0
    # Materialize the list before editing prim types.
    for prim in list(stage.Traverse()):
        if not str(prim.GetPath()).startswith(ROBOT_PATH + "/"):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.FixedJoint.Define(stage, prim.GetPath())
            if not joint or not joint.GetBody1Rel().GetTargets():
                raise RuntimeError(f"Cannot convert G1 joint to FixedJoint: {prim.GetPath()}")
            count += 1
    if count < 20:
        raise RuntimeError(
            f"Only {count} articulated G1 joints found. "
            "Check the G1_29DOF_CFG USD model and its loaded joint prims."
        )
    return count


def _find_right_wrist(stage: Usd.Stage) -> Usd.Prim:
    candidates = [
        prim for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(ROBOT_PATH + "/")
        and prim.GetName() == "right_wrist_yaw_link"
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one rigid right_wrist_yaw_link, found {len(candidates)}. "
            "Inspect the imported G1 USD hierarchy."
        )
    return candidates[0]


def _set_color(gprim: UsdGeom.Gprim, rgb: tuple[float, float, float]) -> None:
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])


def _create_pendulum(stage: Usd.Stage, wrist: Usd.Prim) -> Gf.Vec3d:
    """Create a massive cylinder and a truly free (unlimited) 3-DOF joint."""
    wrist_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(wrist)
    pivot_world = wrist_world.Transform(Gf.Vec3d(*PIVOT_IN_WRIST))

    # Rotation about world axes: +Y tilts the rod toward world +X.
    tilt_x = Gf.Rotation(Gf.Vec3d(1, 0, 0), args.tilt_x_deg)
    tilt_y = Gf.Rotation(Gf.Vec3d(0, 1, 0), args.tilt_y_deg)
    rod_orientation = tilt_y * tilt_x
    center_world = pivot_world + rod_orientation.TransformDir(Gf.Vec3d(0, 0, ROD_LENGTH_M / 2))

    UsdGeom.Xform.Define(stage, "/World/Pendulum")
    rod = UsdGeom.Cylinder.Define(stage, ROD_PATH)
    rod.CreateAxisAttr("Z")
    rod.CreateRadiusAttr(ROD_RADIUS_M)
    rod.CreateHeightAttr(ROD_LENGTH_M)
    _set_color(rod, (0.86, 0.87, 0.90))
    xform = UsdGeom.Xformable(rod.GetPrim())
    xform.AddTranslateOp().Set(center_world)
    xform.AddOrientOp().Set(rod_orientation.GetQuat())

    UsdPhysics.RigidBodyAPI.Apply(rod.GetPrim())
    # Do NOT add CollisionAPI: G1 / ground interference is explicitly excluded.
    total_mass = ROD_MASS_KG + ATOM_MASS_KG
    mass_api = UsdPhysics.MassAPI.Apply(rod.GetPrim())
    mass_api.CreateMassAttr(total_mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    # Uniform solid cylinder + centered cubic Atom. Both share the same COM.
    i_transverse = (
        ROD_MASS_KG * (3 * ROD_RADIUS_M**2 + ROD_LENGTH_M**2) / 12
        + ATOM_MASS_KG * (ATOM_SIDE_M**2 + ATOM_SIDE_M**2) / 12
    )
    i_axial = (
        ROD_MASS_KG * ROD_RADIUS_M**2 / 2
        + ATOM_MASS_KG * (ATOM_SIDE_M**2 + ATOM_SIDE_M**2) / 6
    )
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(i_transverse, i_transverse, i_axial))

    # Visual representation only: its 20g mass is already included above.
    atom = UsdGeom.Cube.Define(stage, ROD_PATH + "/AtomVisual")
    atom.CreateSizeAttr(ATOM_SIDE_M)
    _set_color(atom, (0.12, 0.23, 0.85))

    joint = UsdPhysics.SphericalJoint.Define(stage, JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([wrist.GetPath()])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ROD_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*PIVOT_IN_WRIST))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -ROD_LENGTH_M / 2))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    # No cone limits / drives: rotation is completely free.
    return pivot_world


def _rod_direction(stage: Usd.Stage) -> tuple[float, float, float]:
    # use_fabric=False lets PhysX update transforms in USD each physics step.
    rod = stage.GetPrimAtPath(ROD_PATH)
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(rod)
    direction = transform.TransformDir(Gf.Vec3d(0, 0, 1))
    magnitude = direction.GetLength()
    if magnitude == 0:
        raise RuntimeError("Rod direction has zero length.")
    return tuple(float(v / magnitude) for v in direction)


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=1 / 240, use_fabric=False))
    sim.set_camera_view(eye=[2.5, -2.5, 1.8], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=1800.0)
    light.func("/World/Light", light)

    # Spawn a real Isaac Lab G1 USD model, but do not instantiate an Articulation
    # controller. All robot DOFs are replaced by fixed USD physics joints.
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    robot_cfg.spawn.func(
        ROBOT_PATH, robot_cfg.spawn,
        translation=robot_cfg.init_state.pos, orientation=robot_cfg.init_state.rot,
    )
    stage = sim_utils.get_current_stage()
    locked_joints = _lock_robot_joints(stage)
    wrist = _find_right_wrist(stage)
    pivot = _create_pendulum(stage, wrist)
    print(f"[INFO] Locked G1 joints: {locked_joints}")
    print(f"[INFO] Right wrist: {wrist.GetPath()}")
    print(f"[INFO] Pivot (world, m): {tuple(pivot)}")
    print(f"[INFO] Rod 1.0 m; initial X/Y tilts: {args.tilt_x_deg}/{args.tilt_y_deg} deg")
    print("[INFO] Spherical joint: 3 free rotations, no angle limits, no drives.")
    print("[INFO] Pendulum collision disabled; only gravitational motion is tested.")

    # IMPORTANT: construct all constraints BEFORE PhysX starts.
    sim.reset()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    max_tilt = 0.0
    initial_tilt = rod_tilt_deg(_rod_direction(stage))[0]
    dt = sim.get_physics_dt()
    steps = math.ceil(args.duration / dt)

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "tilt_deg", "tilt_toward_x_deg", "tilt_toward_y_deg", "unit_x", "unit_y", "unit_z"])
        for step in range(steps + 1):
            if step and not simulation_app.is_running():
                break
            if step:
                sim.step()
            ux, uy, uz = _rod_direction(stage)
            tilt, pitch_x, pitch_y = rod_tilt_deg((ux, uy, uz))
            max_tilt = max(max_tilt, tilt)
            if step % args.print_every == 0:
                print(f"t={step * dt:6.3f}s  tilt={tilt:7.2f}deg   ux={ux:+.3f} uy={uy:+.3f} uz={uz:+.3f}")
            writer.writerow([f"{step * dt:.6f}", f"{tilt:.6f}", f"{pitch_x:.6f}", f"{pitch_y:.6f}", ux, uy, uz])

    print(f"[RESULT] Initial tilt: {initial_tilt:.2f} deg; maximum tilt: {max_tilt:.2f} deg")
    print(f"[RESULT] CSV: {args.csv.resolve()}")
    if args.assert_fall and max_tilt < initial_tilt + 2.0:
        raise RuntimeError("Fall assertion failed: tilt did not increase by 2 degrees. Check PhysX warnings or joint setup.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
