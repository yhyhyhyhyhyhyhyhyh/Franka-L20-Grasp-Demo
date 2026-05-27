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
LOG_PATH = PROJECT_DIR / "logs/l20_action_mapping_reward_probe.txt"

BALL_POSITION = np.array([0.32, 0.0, 0.50], dtype=np.float64)
BALL_RADIUS = 0.045

FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]

HOME_Q = np.array([0.0, -0.65, 0.0, -2.10, 0.0, 1.65, 0.75], dtype=np.float64)

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
    joint.CreateLocalRot0Attr(Gf.Quatf(float(local_rot0[0]), Gf.Vec3f(float(local_rot0[1]), float(local_rot0[2]), float(local_rot0[3]))))
    joint.CreateLocalRot1Attr(Gf.Quatf(float(local_rot1[0]), Gf.Vec3f(float(local_rot1[1]), float(local_rot1[2]), float(local_rot1[3]))))
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
        drive.CreateStiffnessAttr(1600.0)
        drive.CreateDampingAttr(80.0)
        drive.CreateMaxForceAttr(180.0)
    return out


def set_l20_targets(joint_prims, target_rad):
    for name, prim in joint_prims.items():
        val = float(target_rad.get(name, 0.0))
        lo, hi = L20_JOINT_LIMITS[name]
        val = max(lo, min(hi, val))
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetTargetPositionAttr().Set(float(math.degrees(val)))


def action8_to_l20_targets(action):
    """
    action: 8维，范围建议 [-1, 1]
    这里先转成 [0, 1] 的 closing amount，再映射到关节角。
    """
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    u = 0.5 * (a + 1.0)

    target = {k: 0.0 for k in L20_JOINT_LIMITS.keys()}

    # a0 index mcp pitch
    target["index_mcp_pitch"] = 1.05 * u[0]

    # a1 middle/ring/pinky mcp pitch
    for f in ["middle", "ring", "pinky"]:
        target[f"{f}_mcp_pitch"] = 1.05 * u[1]

    # a2 index pip/dip
    target["index_pip"] = 1.20 * u[2]
    target["index_dip"] = 0.85 * u[2]

    # a3 middle/ring/pinky pip/dip
    for f in ["middle", "ring", "pinky"]:
        target[f"{f}_pip"] = 1.20 * u[3]
        target[f"{f}_dip"] = 0.85 * u[3]

    # a4 thumb yaw
    target["thumb_cmc_yaw"] = 1.25 * u[4]

    # a5 thumb roll
    target["thumb_cmc_roll"] = 1.05 * u[5]

    # a6 thumb pitch
    target["thumb_cmc_pitch"] = 0.75 * u[6]

    # a7 thumb flexion
    target["thumb_mcp"] = 0.95 * u[7]
    target["thumb_dip"] = 1.05 * u[7]

    return target


def compute_reward_terms(stage):
    ball = BALL_POSITION.copy()
    tips = {}
    for name, path in FINGERTIP_PATHS.items():
        tips[name] = get_world_pos(stage, path)

    palm = get_avg_pos(stage, PALM_PATHS)

    dists = {name: float(np.linalg.norm(pos - ball)) for name, pos in tips.items()}
    four = ["index", "middle", "ring", "pinky"]

    thumb_dist = dists["thumb"]
    four_mean_dist = float(np.mean([dists[k] for k in four]))
    palm_dist = float(np.linalg.norm(palm - ball))

    thumb_vec = tips["thumb"] - ball
    four_center = np.mean(np.stack([tips[k] for k in four], axis=0), axis=0)
    four_vec = four_center - ball

    def norm(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v * 0

    opposition_dot = float(np.dot(norm(thumb_vec), norm(four_vec)))

    num_four_close = sum(1 for k in four if dists[k] < 0.075)
    thumb_close = thumb_dist < 0.070
    opposition_ok = opposition_dot < -0.20

    # 仿 UniDexGrasp2：指尖距离 + 手掌距离 + 成功 bonus
    reward = (
        -3.0 * thumb_dist
        -1.0 * four_mean_dist
        -0.4 * palm_dist
        + 1.0 * num_four_close
        + (3.0 if thumb_close else 0.0)
        + (2.0 if opposition_ok else 0.0)
        + (5.0 if (thumb_close and num_four_close >= 3 and opposition_ok) else 0.0)
    )

    return {
        "reward": float(reward),
        "thumb_dist": float(thumb_dist),
        "four_mean_dist": float(four_mean_dist),
        "palm_dist": float(palm_dist),
        "opposition_dot": float(opposition_dot),
        "num_four_close": int(num_four_close),
        "thumb_close": bool(thumb_close),
        "opposition_ok": bool(opposition_ok),
        "success_like": bool(thumb_close and num_four_close >= 3 and opposition_ok),
        "tip_dists": dists,
        "tips": {k: v.tolist() for k, v in tips.items()},
        "palm": palm.tolist(),
        "ball": ball.tolist(),
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

    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_hand/panda_finger_joint1")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_hand/panda_finger_joint2")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_leftfinger")
    delete_prim_if_exists(stage, "/World/Robot/Franka/panda_rightfinger")
    delete_prim_if_exists(stage, f"{L20_ROOT}/root_joint")

    create_fixed_joint(stage, "/World/Robot/Franka_Base_Fixed_To_World", None, "/World/Robot/Franka/panda_link0")
    create_fixed_joint(
        stage, "/World/Robot/Franka_L20_Real_FixedJoint",
        "/World/Robot/Franka/panda_hand",
        f"{L20_ROOT}/base_link",
        local_pos0=(0.0, 0.0, 0.08),
        local_pos1=(0.0, 0.0, 0.0),
        local_rot0=(0.7071068, 0.0, 0.0, -0.7071068),
        local_rot1=(1.0, 0.0, 0.0, 0.0),
    )

    create_static_collision_sphere(stage, BALL_PATH, BALL_POSITION, BALL_RADIUS)

    if not stage.GetPrimAtPath("/World/physicsScene").IsValid():
        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)

    return stage


def main():
    with open(PREGRASP_JSON, "r") as f:
        pregrasp = json.load(f)
    target_q = np.array(pregrasp["q_rad"], dtype=np.float64)

    stage = setup_scene()
    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()

    franka_joints = setup_franka_drives(stage)
    l20_joints = setup_l20_drives(stage)

    set_franka_q(franka_joints, HOME_Q)
    set_l20_targets(l20_joints, action8_to_l20_targets([-1]*8))
    for _ in range(120):
        sim.step(render=True)

    print("\n========== 19 L20 ACTION MAPPING REWARD PROBE ==========")
    print("This is NOT PPO yet. It verifies fingertip paths, 8D action mapping, and reward terms.")
    print("Target q:", target_q.tolist())

    for alpha in np.linspace(0, 1, 240):
        set_franka_q(franka_joints, (1-alpha)*HOME_Q + alpha*target_q)
        set_l20_targets(l20_joints, action8_to_l20_targets([-1]*8))
        sim.step(render=True)

    tests = [
        ("open",              [-1, -1, -1, -1, -1, -1, -1, -1]),
        ("four_fingers_only", [ 0.8, 0.8, 0.8, 0.8, -1, -1, -1, -1]),
        ("thumb_only",        [-1, -1, -1, -1, 0.8, 0.8, 0.8, 0.8]),
        ("all_medium",        [ 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        ("all_strong",        [ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
        ("thumb_more",        [ 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
    ]

    results = []

    for name, action in tests:
        print(f"\n>>> TEST: {name}, action={action}")

        target = action8_to_l20_targets(action)
        set_l20_targets(l20_joints, target)

        for _ in range(220):
            set_franka_q(franka_joints, target_q)
            sim.step(render=True)

        terms = compute_reward_terms(stage)
        results.append({
            "name": name,
            "action": action,
            "targets_rad": target,
            "terms": terms,
        })

        print(json.dumps(terms, indent=2))

    with open(LOG_PATH, "w") as f:
        json.dump({
            "pregrasp_json": str(PREGRASP_JSON),
            "ball_position": BALL_POSITION.tolist(),
            "ball_radius": BALL_RADIUS,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved log to: {LOG_PATH}")
    print("Inspect final pose. Press Ctrl+C when done.")

    while simulation_app.is_running():
        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
