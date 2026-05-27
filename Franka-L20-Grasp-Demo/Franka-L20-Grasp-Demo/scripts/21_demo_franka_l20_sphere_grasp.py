import json
import math
from pathlib import Path
import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd

try:
    from isaacsim.core.api import SimulationContext
except Exception:
    from omni.isaac.core import SimulationContext


# ============================================================
# Paths
# ============================================================

FRANKA_USD_PATH = "/opt/IsaacSimAssets/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
L20_USD_PATH = "/home/glm01emy/Desktop/dexhand_assets/L20_right/linkerhand_l20_right/linkerhand_l20_right.usd"

WORLD_PATH = "/World"
ROBOT_PATH = "/World/Robot"
FRANKA_PATH = "/World/Robot/Franka"
L20_PATH = "/World/Robot/L20"
L20_ROOT = "/World/Robot/L20/linkerhand_l20_right"
BALL_PATH = "/World/Ball"

PROJECT_DIR = Path("/home/glm01emy/Desktop/franka_l20_cup_pick_place")
PREGRASP_JSON = PROJECT_DIR / "logs/manual_selected_pregrasp_sphere.json"
LOG_PATH = PROJECT_DIR / "logs/demo_franka_l20_sphere_grasp.txt"

BALL_POSITION = np.array([0.32, 0.0, 0.50], dtype=np.float64)
BALL_RADIUS = 0.045


# ============================================================
# Franka joints
# ============================================================

FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

HOME_Q = np.array([
    0.0,
    -0.65,
    0.0,
    -2.10,
    0.0,
    1.65,
    0.75,
], dtype=np.float64)


# ============================================================
# L20 joints
# ============================================================

L20_JOINT_LIMITS = {
    "pinky_mcp_roll": (-0.17, 0.17),
    "pinky_mcp_pitch": (0.0, 1.4),
    "pinky_pip": (0.0, 1.57),
    "pinky_dip": (0.0, 1.4),

    "ring_mcp_roll": (-0.17, 0.17),
    "ring_mcp_pitch": (0.0, 1.4),
    "ring_pip": (0.0, 1.57),
    "ring_dip": (0.0, 1.4),

    "middle_mcp_roll": (-0.17, 0.17),
    "middle_mcp_pitch": (0.0, 1.4),
    "middle_pip": (0.0, 1.57),
    "middle_dip": (0.0, 1.4),

    "index_mcp_roll": (-0.17, 0.17),
    "index_mcp_pitch": (0.0, 1.4),
    "index_pip": (0.0, 1.57),
    "index_dip": (0.0, 1.4),

    "thumb_cmc_yaw": (0.0, 1.4),
    "thumb_cmc_roll": (0.0, 1.22),
    "thumb_cmc_pitch": (0.0, 0.79),
    "thumb_mcp": (0.0, 1.05),
    "thumb_dip": (0.0, 1.22),
}


# all_strong 动作，对应刚才 reward probe 中视觉和数值都比较好的闭合姿态
L20_FINAL_TARGET_RAD = {
    "pinky_mcp_roll": 0.0,
    "pinky_mcp_pitch": 0.9975,
    "pinky_pip": 1.14,
    "pinky_dip": 0.8075,

    "ring_mcp_roll": 0.0,
    "ring_mcp_pitch": 0.9975,
    "ring_pip": 1.14,
    "ring_dip": 0.8075,

    "middle_mcp_roll": 0.0,
    "middle_mcp_pitch": 0.9975,
    "middle_pip": 1.14,
    "middle_dip": 0.8075,

    "index_mcp_roll": 0.0,
    "index_mcp_pitch": 0.9975,
    "index_pip": 1.14,
    "index_dip": 0.8075,

    "thumb_cmc_yaw": 1.1875,
    "thumb_cmc_roll": 0.9975,
    "thumb_cmc_pitch": 0.7125,
    "thumb_mcp": 0.9025,
    "thumb_dip": 0.9975,
}


FINGERTIP_PATHS = {
    "index": f"{L20_ROOT}/index_distal",
    "middle": f"{L20_ROOT}/middle_distal",
    "ring": f"{L20_ROOT}/ring_distal",
    "pinky": f"{L20_ROOT}/pinky_distal",
    "thumb": f"{L20_ROOT}/thumb_distal",
}

PALM_PATHS = [
    f"{L20_ROOT}/index_proximal",
    f"{L20_ROOT}/middle_proximal",
    f"{L20_ROOT}/ring_proximal",
    f"{L20_ROOT}/pinky_proximal",
]


# ============================================================
# Utils
# ============================================================

def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def delete_prim_if_exists(stage, path):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)


def create_xform_with_reference(stage, path, usd_path):
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    prim.GetReferences().AddReference(usd_path)
    return prim


def create_fixed_joint(stage, joint_path, body0_path, body1_path,
                       local_pos0=(0, 0, 0), local_pos1=(0, 0, 0),
                       local_rot0=(1, 0, 0, 0), local_rot1=(1, 0, 0, 0)):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    if body0_path is not None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    if body1_path is not None:
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local_pos1))

    joint.CreateLocalRot0Attr(Gf.Quatf(
        float(local_rot0[0]),
        Gf.Vec3f(float(local_rot0[1]), float(local_rot0[2]), float(local_rot0[3]))
    ))
    joint.CreateLocalRot1Attr(Gf.Quatf(
        float(local_rot1[0]),
        Gf.Vec3f(float(local_rot1[1]), float(local_rot1[2]), float(local_rot1[3]))
    ))

    return joint


def create_static_collision_sphere(stage, path, position, radius):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)

    xform_api = UsdGeom.XformCommonAPI(sph.GetPrim())
    xform_api.SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    return sph.GetPrim()


def get_world_pos(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {path}")
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def get_avg_pos(stage, paths):
    return np.mean(np.stack([get_world_pos(stage, p) for p in paths], axis=0), axis=0)


def find_joint_prim(stage, joint_name):
    for prim in stage.Traverse():
        if prim.GetName() == joint_name:
            return prim
    raise RuntimeError(f"Cannot find joint prim: {joint_name}")


def setup_franka_drives(stage):
    out = []
    for name in FRANKA_JOINT_NAMES:
        prim = find_joint_prim(stage, name)
        out.append(prim)

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(9000.0)
        drive.CreateDampingAttr(350.0)
        drive.CreateMaxForceAttr(2500.0)

    return out


def set_franka_q(joint_prims, q_rad):
    for prim, val in zip(joint_prims, np.rad2deg(q_rad)):
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetTargetPositionAttr().Set(float(val))


def setup_l20_drives(stage):
    out = {}

    for name in L20_JOINT_LIMITS:
        prim = find_joint_prim(stage, name)
        out[name] = prim

        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(1800.0)
        drive.CreateDampingAttr(90.0)
        drive.CreateMaxForceAttr(220.0)

    return out


def set_l20_targets(joint_prims, target_rad):
    for name, prim in joint_prims.items():
        val = float(target_rad.get(name, 0.0))
        lo, hi = L20_JOINT_LIMITS[name]
        val = max(lo, min(hi, val))

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetTargetPositionAttr().Set(float(math.degrees(val)))


def interpolate_l20_targets(ratio):
    ratio = smoothstep(ratio)
    target = {}
    for name in L20_JOINT_LIMITS:
        target[name] = ratio * L20_FINAL_TARGET_RAD.get(name, 0.0)
    return target


def compute_demo_metrics(stage):
    ball = BALL_POSITION.copy()
    tips = {}
    for name, path in FINGERTIP_PATHS.items():
        try:
            tips[name] = get_world_pos(stage, path)
        except Exception:
            tips[name] = np.array([np.nan, np.nan, np.nan])

    palm = get_avg_pos(stage, PALM_PATHS)

    tip_dists = {
        name: float(np.linalg.norm(pos - ball))
        for name, pos in tips.items()
    }

    four_mean = float(np.mean([
        tip_dists["index"],
        tip_dists["middle"],
        tip_dists["ring"],
        tip_dists["pinky"],
    ]))

    return {
        "ball_position": ball.tolist(),
        "palm_position": palm.tolist(),
        "palm_dist": float(np.linalg.norm(palm - ball)),
        "tip_dists": tip_dists,
        "four_mean_dist": four_mean,
        "thumb_dist": tip_dists["thumb"],
    }


def setup_scene():
    stage = omni.usd.get_context().get_stage()

    if not stage.GetPrimAtPath(WORLD_PATH).IsValid():
        UsdGeom.Xform.Define(stage, WORLD_PATH)

    stage.SetDefaultPrim(stage.GetPrimAtPath(WORLD_PATH))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    delete_prim_if_exists(stage, ROBOT_PATH)
    delete_prim_if_exists(stage, BALL_PATH)

    UsdGeom.Xform.Define(stage, ROBOT_PATH)

    create_xform_with_reference(stage, FRANKA_PATH, FRANKA_USD_PATH)
    create_xform_with_reference(stage, L20_PATH, L20_USD_PATH)

    # 删除 Franka 原夹爪
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_hand/panda_finger_joint1")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_hand/panda_finger_joint2")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_leftfinger")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_rightfinger")

    # 删除 L20 原 root joint
    delete_prim_if_exists(stage, f"{L20_ROOT}/root_joint")

    # 固定 Franka 底座
    create_fixed_joint(
        stage,
        "/World/Robot/Franka_Base_Fixed_To_World",
        None,
        "/World/Robot/Franka/panda_link0",
    )

    # 固定 L20 到 Franka
    create_fixed_joint(
        stage,
        "/World/Robot/Franka_L20_Real_FixedJoint",
        "/World/Robot/Franka/panda_hand",
        f"{L20_ROOT}/base_link",
        local_pos0=(0.0, 0.0, 0.08),
        local_pos1=(0.0, 0.0, 0.0),
        local_rot0=(0.7071068, 0.0, 0.0, -0.7071068),
        local_rot1=(1.0, 0.0, 0.0, 0.0),
    )

    create_static_collision_sphere(
        stage,
        BALL_PATH,
        position=BALL_POSITION,
        radius=BALL_RADIUS,
    )

    if not stage.GetPrimAtPath("/World/physicsScene").IsValid():
        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)

    return stage


# ============================================================
# Main
# ============================================================

def main():
    if not PREGRASP_JSON.exists():
        raise FileNotFoundError(f"Cannot find: {PREGRASP_JSON}")

    with open(PREGRASP_JSON, "r") as f:
        pregrasp = json.load(f)

    target_q = np.array(pregrasp["q_rad"], dtype=np.float64)

    stage = setup_scene()

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()

    franka_joints = setup_franka_drives(stage)
    l20_joints = setup_l20_drives(stage)

    print("\n========== 21 DEMO: FRANKA + L20 SPHERE GRASP ==========")
    print("Demo sequence:")
    print("  1. Start from home pose")
    print("  2. Move Franka + L20 to manually selected pre-grasp pose")
    print("  3. Close LinkerHand L20 fingers")
    print("  4. Hold final grasp pose for video recording")
    print("")
    print("This demo uses a STATIC sphere for visual stability.")
    print("It is intended as a project demonstration video, not yet a validated rigid-body lift test.")
    print("")
    print("target_q_rad =", target_q.tolist())
    print("target_q_deg =", np.rad2deg(target_q).tolist())

    # 阶段 0：初始姿态，手张开，保持一会儿
    set_franka_q(franka_joints, HOME_Q)
    set_l20_targets(l20_joints, interpolate_l20_targets(0.0))

    print("\n[Stage 0] Home pose, hand open.")
    for _ in range(180):
        sim.step(render=True)

    # 阶段 1：机械臂从 home 平滑移动到 pregrasp
    print("[Stage 1] Moving to pre-grasp pose.")
    n_move = 360
    for i in range(n_move):
        alpha = smoothstep(i / (n_move - 1))
        q_now = (1.0 - alpha) * HOME_Q + alpha * target_q
        set_franka_q(franka_joints, q_now)
        set_l20_targets(l20_joints, interpolate_l20_targets(0.0))
        sim.step(render=True)

    print("[Stage 1] Arrived pre-grasp. Holding.")
    for _ in range(120):
        set_franka_q(franka_joints, target_q)
        set_l20_targets(l20_joints, interpolate_l20_targets(0.0))
        sim.step(render=True)

    metrics_pre = compute_demo_metrics(stage)
    print("\nPRE-GRASP METRICS:")
    print(json.dumps(metrics_pre, indent=2))

    # 阶段 2：L20 手指平滑闭合
    print("[Stage 2] Closing L20 fingers.")
    n_close = 360
    close_log = []

    for i in range(n_close):
        ratio = smoothstep(i / (n_close - 1))
        set_franka_q(franka_joints, target_q)
        set_l20_targets(l20_joints, interpolate_l20_targets(ratio))
        sim.step(render=True)

        if i % 60 == 0:
            m = compute_demo_metrics(stage)
            close_log.append({"frame": i, "close_ratio": ratio, **m})
            print(
                f"close_ratio={ratio:.3f}, "
                f"thumb_dist={m['thumb_dist']:.4f}, "
                f"four_mean={m['four_mean_dist']:.4f}, "
                f"palm_dist={m['palm_dist']:.4f}"
            )

    # 阶段 3：最终抓握姿态保持
    print("[Stage 3] Holding final grasp pose.")
    for _ in range(600):
        set_franka_q(franka_joints, target_q)
        set_l20_targets(l20_joints, interpolate_l20_targets(1.0))
        sim.step(render=True)

    metrics_final = compute_demo_metrics(stage)

    result = {
        "pregrasp_json": str(PREGRASP_JSON),
        "ball_position": BALL_POSITION.tolist(),
        "ball_radius": BALL_RADIUS,
        "home_q_rad": HOME_Q.tolist(),
        "target_q_rad": target_q.tolist(),
        "target_q_deg": np.rad2deg(target_q).tolist(),
        "l20_final_target_rad": L20_FINAL_TARGET_RAD,
        "metrics_pre": metrics_pre,
        "metrics_final": metrics_final,
        "close_log": close_log,
        "note": "Static sphere demo for stable video recording. Not yet rigid-body lift validation.",
    }

    with open(LOG_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n========== DEMO FINAL ==========")
    print(json.dumps(metrics_final, indent=2))
    print(f"\nSaved demo log to: {LOG_PATH}")
    print("")
    print("Demo is complete. Window will stay open for recording.")
    print("Use screen recording now if needed. Press Ctrl+C in terminal when done.")

    while simulation_app.is_running():
        set_franka_q(franka_joints, target_q)
        set_l20_targets(l20_joints, interpolate_l20_targets(1.0))
        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
