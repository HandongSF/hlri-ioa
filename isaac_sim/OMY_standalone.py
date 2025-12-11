#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interactive OMY-F3M pose logger with ROS2 JointState control."""

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_STAGE_ROOT = Path.home() / "stages" / "OMY_sim"


def parse_args() -> argparse.Namespace:
    """Return CLI arguments so paths/topics can be configured easily."""

    parser = argparse.ArgumentParser(
        description=(
            "Toggle CSV logging from the Isaac Sim GUI while randomizing "
            "target poses and feeding ROS2 JointState commands."
        )
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=DEFAULT_STAGE_ROOT,
        help=f"Directory that holds OMY assets (default: {DEFAULT_STAGE_ROOT}).",
    )
    parser.add_argument(
        "--usd-path",
        type=Path,
        default=None,
        help="Path to the robot USD file (default: <stage-root>/OMY.usd).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory used for CSV logs (default: <stage-root>/logs).",
    )
    parser.add_argument(
        "--joint-topic",
        default="/joint_states",
        help="ROS2 JointState topic name (default: /joint_states).",
    )
    parser.add_argument(
        "--run-start-index",
        type=int,
        default=0,
        help="Initial run index so CSV names can continue existing logs.",
    )
    return parser.parse_args()


args = parse_args()
stage_root = args.stage_root.expanduser()
omy_usd_path = args.usd_path.expanduser() if args.usd_path else stage_root / "OMY.usd"
log_dir = args.log_dir.expanduser() if args.log_dir else stage_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

LOG_DIR = log_dir
OMY_F3M_USD = omy_usd_path
ROS_JOINT_CMD_TOPIC = args.joint_topic

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Gf, UsdLux
from omni.usd import get_context

import omni.graph.core as og
import omni.ui as ui
import omni.kit.app as kit

assets_root = get_assets_root_path()
if assets_root is None:
    print("[ERROR] Isaac Sim assets root not found.")
    simulation_app.close()
    raise SystemExit(1)

ROBOT_ROOT_PRIM = "/World/OMY/OMY"
EE_PRIM_PATH = "/World/OMY/OMY/link6/end_effector_flange_link/end_effector_link"
TARGET_PRIM_PATH = "/World/target_cube"

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 400.0,
)

world.scene.add_default_ground_plane()

print(f"[INFO] OMY_F3M_USD: {OMY_F3M_USD}")

add_reference_to_stage(usd_path=str(OMY_F3M_USD), prim_path="/World/OMY")

TARGET_SCALE = np.array([0.03, 0.03, 0.03], dtype=np.float32)
TARGET_HALF_EXTENTS = TARGET_SCALE * 0.5

cube_prim = prim_utils.create_prim(
    prim_path=TARGET_PRIM_PATH,
    prim_type="Cube",
    position=np.array([0.4, 0.0, 0.3 / 2.0], dtype=np.float32),
    scale=TARGET_SCALE,
)
UsdGeom.Mesh(cube_prim).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

world.reset()

ee_view = XFormPrim(prim_paths_expr=EE_PRIM_PATH, name="ee_view")
target_view = XFormPrim(prim_paths_expr=TARGET_PRIM_PATH, name="target_view")

print("[INFO] EE_PRIM_PATH     :", EE_PRIM_PATH)
print("[INFO] TARGET_PRIM_PATH :", TARGET_PRIM_PATH)

stage = get_context().get_stage()
CAMERA_PATH = "/World/OMY/OMY/link6/Camera"

cam_prim = UsdGeom.Camera.Get(stage, CAMERA_PATH)
if not cam_prim:
    cam_prim = UsdGeom.Camera.Define(stage, CAMERA_PATH)

cam_xform = UsdGeom.Xformable(cam_prim)
for op in cam_xform.GetOrderedXformOps():
    cam_xform.RemoveXformOp(op)

t_op = cam_xform.AddTranslateOp()
t_op.Set(Gf.Vec3f(0.0, -0.1, 0.07))

q = Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.57358, 0.81915))
o_op = cam_xform.AddOrientOp()
o_op.Set(q)

cam_prim.GetFocalLengthAttr().Set(18.14)
cam_prim.GetFocusDistanceAttr().Set(400.0)
cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1_000_000.0))

def randomize_target_pose():
    """Align the cube with the EE pose and add bounded noise."""

    _, ee_quat0 = ee_view.get_world_poses()
    ee_quat0 = ee_quat0[0]
    ee_rpy0 = quat_to_euler_angles(ee_quat0, degrees=False)

    rand_x = float(np.random.uniform(0.4, 0.45))
    rand_y = float(np.random.uniform(-0.3, 0.3))
    rand_z = float(np.random.uniform(0.5, 0.7))
    target_pos = np.array([[rand_x, rand_y, rand_z]], dtype=np.float32)

    roll_offset = float(np.random.uniform(-np.pi / 8.0, np.pi / 8.0))
    pitch_offset = float(np.random.uniform(-np.pi / 8.0, np.pi / 8.0))
    yaw_offset = float(np.random.uniform(-np.pi / 8.0, np.pi / 8.0))

    target_roll = float(ee_rpy0[0] + roll_offset)
    target_pitch = float(ee_rpy0[1] + pitch_offset)
    target_yaw = float(ee_rpy0[2] + yaw_offset)

    euler_ypr = np.array([target_yaw, target_pitch, target_roll], dtype=np.float32)
    quat_ypr = euler_angles_to_quat(euler_ypr, degrees=False)
    target_quat = np.array([quat_ypr], dtype=np.float32)

    target_view.set_world_poses(target_pos, target_quat)

    print("[INFO] Randomized target cube pose:")
    print(f"       position         = {target_pos[0]}")
    print(f"       EE rpy (rad)     = roll={ee_rpy0[0]:.3f}, pitch={ee_rpy0[1]:.3f}, yaw={ee_rpy0[2]:.3f}")
    print(f"       offsets (rad)    = dR={roll_offset:.3f}, dP={pitch_offset:.3f}, dY={yaw_offset:.3f}")
    print(f"       target rpy (rad) = roll={target_roll:.3f}, pitch={target_pitch:.3f}, yaw={target_yaw:.3f}")

randomize_target_pose()


stage = world.stage

EE_MARKER_PATH = EE_PRIM_PATH + "/ee_marker"
ee_marker = UsdGeom.Sphere.Define(stage, EE_MARKER_PATH)
ee_marker.CreateRadiusAttr(0.01)
ee_marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])

print(f"[INFO] EE marker sphere created at {EE_MARKER_PATH}")

light_path = "/World/KeyLight"
key_light = UsdLux.DistantLight.Define(stage, light_path)
key_light.CreateIntensityAttr(5000.0)

print("[INFO] Distant light added at /World/KeyLight")


ext_manager = kit.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
print("[INFO] Enabled extension: isaacsim.ros2.bridge")

def setup_ros2_joint_control_graph(graph_path="/ROS2JointGraph"):
    """Connect ROS2 JointState commands to the articulation controller."""

    print("[INFO] Setting up ROS2 JointState control graph at", graph_path)

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPhysicsStep",         "isaacsim.core.nodes.OnPhysicsStep"),
                ("SubscribeJointState",   "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController","isaacsim.core.nodes.IsaacArticulationController"),
            ],

            og.Controller.Keys.CONNECT: [
                ("OnPhysicsStep.outputs:step", "SubscribeJointState.inputs:execIn"),
                ("OnPhysicsStep.outputs:step", "ArticulationController.inputs:execIn"),

                ("SubscribeJointState.outputs:jointNames",
                 "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand",
                 "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand",
                 "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",
                 "ArticulationController.inputs:effortCommand"),
            ],

            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", ROBOT_ROOT_PRIM),
                ("SubscribeJointState.inputs:topicName", ROS_JOINT_CMD_TOPIC),
            ],
        },
    )

    print("[INFO] ROS2 JointState control graph created.")
    print("[INFO] Pipeline Stage: ONDEMAND (required for OnPhysicsStep)")
    print(f"[INFO] Subscribing JointState commands from topic: {ROS_JOINT_CMD_TOPIC}")

setup_ros2_joint_control_graph()

fieldnames = [
    "t_sim",
    "ee_x", "ee_y", "ee_z",
    "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "ee_roll", "ee_pitch", "ee_yaw",
    "target_x", "target_y", "target_z",
    "target_qw", "target_qx", "target_qy", "target_qz",
    "target_roll", "target_pitch", "target_yaw",
    "err_x", "err_y", "err_z",
    "err_dist",
    "err_roll", "err_pitch", "err_yaw"
]

csv_file = None
writer = None

logging_active = False
logging_start_sim_time = None
run_index = args.run_start_index

status_label = None
toggle_button = None

def stop_logging(reason: str = ""):
    """Common logic to stop logging and close CSV."""
    global logging_active, logging_start_sim_time, csv_file, writer, status_label, toggle_button, run_index

    if not logging_active:
        return

    if reason:
        print(f"[INFO] LOG STOP (run {run_index:03d}) – reason: {reason}")
    else:
        print(f"[INFO] LOG STOP (run {run_index:03d})")

    if csv_file is not None:
        csv_file.flush()
        csv_file.close()
        csv_file = None
        writer = None

    logging_active = False
    logging_start_sim_time = None

    if status_label:
        status_label.text = "Status: IDLE"
    if toggle_button:
        toggle_button.text = "Start Logging"
    randomize_target_pose()

def ee_inside_target_cube(
    ee_pos_world: np.ndarray,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
) -> bool:
    """Return True when the EE marker center lies inside the rotated cube."""

    diff = ee_pos_world - target_pos_world
    w, x, y, z = (
        float(target_quat_world[0]),
        float(target_quat_world[1]),
        float(target_quat_world[2]),
        float(target_quat_world[3]),
    )

    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)

    p_local = R.T.dot(diff)

    return (
        abs(p_local[0]) <= TARGET_HALF_EXTENTS[0] and
        abs(p_local[1]) <= TARGET_HALF_EXTENTS[1] and
        abs(p_local[2]) <= TARGET_HALF_EXTENTS[2]
    )

def toggle_logging():
    """UI button callback: toggle logging on/off"""
    global logging_active, logging_start_sim_time, run_index
    global csv_file, writer, status_label, toggle_button

    if not logging_active:
        run_index += 1
        log_path = LOG_DIR / f"omy_imitation_log_run{run_index:03d}.csv"
        csv_file = log_path.open(mode="w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        logging_start_sim_time = world.current_time
        logging_active = True

        print(f"[INFO] LOG START (run {run_index:03d}) -> {log_path}")
        if status_label:
            status_label.text = f"Status: LOGGING (run {run_index:03d})"
        if toggle_button:
            toggle_button.text = "Stop Logging"
    else:
        stop_logging(reason="button pressed")

window = ui.Window("OMY Imitation Logger", width=280, height=120)

with window.frame:
    with ui.VStack():
        status_label = ui.Label("Status: IDLE")
        toggle_button = ui.Button("Start Logging", clicked_fn=toggle_logging)
        ui.Label(
            "- Click to start/stop CSV logging\n"
            "- Each time you start, a new file (runXXX) is created.",
            word_wrap=True,
        )


try:
    while simulation_app.is_running():
        world.step(render=True)

        if logging_active and writer is not None and logging_start_sim_time is not None:
            t_sim_raw = world.current_time
            t_sim = t_sim_raw - logging_start_sim_time

            ee_pos, ee_quat = ee_view.get_world_poses()
            tgt_pos, tgt_quat = target_view.get_world_poses()

            ee_pos = ee_pos[0]
            ee_quat = ee_quat[0]
            tgt_pos = tgt_pos[0]
            tgt_quat = tgt_quat[0]

            ee_rpy = quat_to_euler_angles(ee_quat, degrees=False)
            tgt_rpy = quat_to_euler_angles(tgt_quat, degrees=False)

            pos_err = tgt_pos - ee_pos
            err_dist = float(np.linalg.norm(pos_err))

            rpy_err = tgt_rpy - ee_rpy
            rpy_err = (rpy_err + np.pi) % (2 * np.pi) - np.pi

            row = {
                "t_sim": float(t_sim),

                "ee_x": float(ee_pos[0]),
                "ee_y": float(ee_pos[1]),
                "ee_z": float(ee_pos[2]),
                "ee_qw": float(ee_quat[0]),
                "ee_qx": float(ee_quat[1]),
                "ee_qy": float(ee_quat[2]),
                "ee_qz": float(ee_quat[3]),
                "ee_roll": float(ee_rpy[0]),
                "ee_pitch": float(ee_rpy[1]),
                "ee_yaw": float(ee_rpy[2]),

                "target_x": float(tgt_pos[0]),
                "target_y": float(tgt_pos[1]),
                "target_z": float(tgt_pos[2]),
                "target_qw": float(tgt_quat[0]),
                "target_qx": float(tgt_quat[1]),
                "target_qy": float(tgt_quat[2]),
                "target_qz": float(tgt_quat[3]),
                "target_roll": float(tgt_rpy[0]),
                "target_pitch": float(tgt_rpy[1]),
                "target_yaw": float(tgt_rpy[2]),

                "err_x": float(pos_err[0]),
                "err_y": float(pos_err[1]),
                "err_z": float(pos_err[2]),
                "err_dist": err_dist,
                "err_roll": float(rpy_err[0]),
                "err_pitch": float(rpy_err[1]),
                "err_yaw": float(rpy_err[2]),
            }

            writer.writerow(row)
            if ee_inside_target_cube(ee_pos, tgt_pos, tgt_quat):
                stop_logging(reason="EE marker inside target cube")

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt: Logging stopped by user.")

finally:
    if csv_file is not None:
        csv_file.flush()
        csv_file.close()
    world.stop()
    simulation_app.close()
    print("[INFO] Simulation closed. CSV file (last run) saved.")
