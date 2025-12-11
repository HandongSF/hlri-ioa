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
"""Evaluate a trained EE-delta BC policy using recorded demonstration logs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

DEFAULT_STAGE_ROOT = Path.home() / "stages" / "OMY_sim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a behavioral cloning policy rollout.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_STAGE_ROOT / "logs" / "omy_imitation_log_run001.csv",
        help="Demo log CSV to evaluate against (default: ~/stages/OMY_sim/logs/omy_imitation_log_run001.csv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_STAGE_ROOT / "ee_bc_policy.pt",
        help="Trained checkpoint path (default: ~/stages/OMY_sim/ee_bc_policy.pt).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum rollout steps (default: 500).",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to Δpose predictions before integration (default: 1.0).",
    )
    parser.add_argument(
        "--pos-tol",
        type=float,
        default=0.05,
        help="Position tolerance (meters) for declaring success (default: 0.05).",
    )
    parser.add_argument(
        "--ang-tol-deg",
        type=float,
        default=5.0,
        help="Angular tolerance (degrees) for declaring success (default: 5°).",
    )
    parser.add_argument(
        "--one-step-eval",
        action="store_true",
        help="Enable one-step prediction RMSE analysis before the rollout.",
    )
    return parser.parse_args()


class BCPolicy(nn.Module):
    def __init__(self, in_dim=6, out_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a2, a1):
    """Return wrapped difference a2 - a1, matching dataset preprocessing."""
    d = a2 - a1
    return angle_wrap(d)


args = parse_args()
CSV_PATH = args.csv.expanduser()
CKPT_PATH = args.checkpoint.expanduser()
MAX_STEPS_ROLLOUT = args.max_steps
ACTION_SCALE = args.action_scale
POS_TOL = args.pos_tol
ANG_TOL = np.deg2rad(args.ang_tol_deg)
DO_ONE_STEP_EVAL = args.one_step_eval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

policy = BCPolicy().to(device)
state = torch.load(CKPT_PATH, map_location=device)

if isinstance(state, dict) and "model_state_dict" in state:
    policy.load_state_dict(state["model_state_dict"])

    obs_mean = torch.tensor(state["obs_mean"], device=device, dtype=torch.float32)
    obs_std = torch.tensor(state["obs_std"], device=device, dtype=torch.float32)

    if "act_mean" in state and "act_std" in state:
        act_mean = torch.tensor(state["act_mean"], device=device, dtype=torch.float32)
        act_std = torch.tensor(state["act_std"], device=device, dtype=torch.float32)
        print("[INFO] act_mean/act_std loaded from checkpoint.")
    else:
        act_mean = torch.zeros((1, 6), device=device, dtype=torch.float32)
        act_std = torch.ones((1, 6), device=device, dtype=torch.float32)
        print("[WARN] act_mean/act_std not found in checkpoint. Using identity (no denorm).")

else:
    policy.load_state_dict(state)
    raise RuntimeError(
        "Checkpoint format mismatch. Please verify the training script output."
    )

policy.eval()
print(f"[INFO] Loaded policy from {CKPT_PATH}")


df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded demo log: {len(df)} steps")

demo_x = df["ee_x"].values
demo_y = df["ee_y"].values
demo_z = df["ee_z"].values

tgt_pos = np.array(
    [
        df["target_x"].values[0],
        df["target_y"].values[0],
        df["target_z"].values[0],
    ],
    dtype=np.float32,
)

tgt_rpy = np.array(
    [
        df["target_roll"].values[0],
        df["target_pitch"].values[0],
        df["target_yaw"].values[0],
    ],
    dtype=np.float32,
)

ee_pos0 = np.array(
    [
        df["ee_x"].values[0],
        df["ee_y"].values[0],
        df["ee_z"].values[0],
    ],
    dtype=np.float32,
)

ee_rpy0 = np.array(
    [
        df["ee_roll"].values[0],
        df["ee_pitch"].values[0],
        df["ee_yaw"].values[0],
    ],
    dtype=np.float32,
)

print("[INFO] Initial EE pos:", ee_pos0)
print("[INFO] Initial EE rpy:", ee_rpy0)
print("[INFO] Target pos:", tgt_pos)
print("[INFO] Target rpy:", tgt_rpy)


if DO_ONE_STEP_EVAL:
    print("\n[DEBUG] Running one-step prediction error on this demo log...")

    obs_cols = ["err_x", "err_y", "err_z", "err_roll", "err_pitch", "err_yaw"]
    errs = []

    for i in range(len(df) - 1):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]

        obs_np = cur[obs_cols].values.astype(np.float32)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        obs_n = (obs_t - obs_mean) / obs_std

        with torch.no_grad():
            pred_norm = policy(obs_n)
            pred = pred_norm * act_std + act_mean
            pred = pred.cpu().numpy()[0]

        dx = nxt["ee_x"] - cur["ee_x"]
        dy = nxt["ee_y"] - cur["ee_y"]
        dz = nxt["ee_z"] - cur["ee_z"]

        droll = angle_diff(nxt["ee_roll"], cur["ee_roll"])
        dpitch = angle_diff(nxt["ee_pitch"], cur["ee_pitch"])
        dyaw = angle_diff(nxt["ee_yaw"], cur["ee_yaw"])

        gt = np.array([dx, dy, dz, droll, dpitch, dyaw], dtype=np.float32)

        errs.append(pred - gt)

    errs = np.stack(errs, axis=0)
    mse_per_dim = np.mean(errs**2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    print("[DEBUG] One-step RMSE per dim [dx,dy,dz,droll,dpitch,dyaw]:")
    print("        ", rmse_per_dim)
    print("[DEBUG] One-step RMSE (position 3D norm):",
          np.mean(np.linalg.norm(errs[:, :3], axis=1)))
    print("[DEBUG] One-step RMSE (angle 3D norm):",
          np.mean(np.linalg.norm(errs[:, 3:], axis=1)))
    print("[DEBUG] Compare with train/val loss values to sanity-check the policy.\n")


max_steps = MAX_STEPS_ROLLOUT
ee_pos = ee_pos0.copy()
ee_rpy = ee_rpy0.copy()

traj_x = [ee_pos[0]]
traj_y = [ee_pos[1]]
traj_z = [ee_pos[2]]

for step in range(max_steps):
    pos_err = tgt_pos - ee_pos
    rpy_err = angle_wrap(tgt_rpy - ee_rpy)

    obs = np.concatenate([pos_err, rpy_err]).astype(np.float32)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    obs_n = (obs_t - obs_mean) / obs_std

    with torch.no_grad():
        d_pose_norm = policy(obs_n)
        d_pose = d_pose_norm * act_std + act_mean
        d_pose = d_pose.cpu().numpy()[0]

    d_pose = d_pose * ACTION_SCALE

    d_pos = d_pose[:3]
    d_rpy = d_pose[3:]

    ee_pos = ee_pos + d_pos
    ee_rpy = angle_wrap(ee_rpy + d_rpy)

    traj_x.append(ee_pos[0])
    traj_y.append(ee_pos[1])
    traj_z.append(ee_pos[2])

    pos_err_norm = np.linalg.norm(tgt_pos - ee_pos)
    ang_err_norm = np.linalg.norm(angle_wrap(tgt_rpy - ee_rpy))

    if step % 50 == 0:
        print(
            f"[ROLLOUT] step={step:04d} | "
            f"pos_err={pos_err_norm:.4f}, ang_err={ang_err_norm:.4f}"
        )

    if pos_err_norm < POS_TOL and ang_err_norm < ANG_TOL:
        print(
            f"[INFO] Policy reached target at step {step}, "
            f"pos_err={pos_err_norm:.4f}, ang_err={ang_err_norm:.4f}"
        )
        break

print(f"[INFO] Policy rollout length: {len(traj_x)}")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

demo_line,  = ax.plot([], [], [], lw=1, alpha=0.5, label="Demo EE path")
demo_point, = ax.plot([], [], [], "o", alpha=0.8, label="Demo current")

policy_line,  = ax.plot([], [], [], lw=2, label="Policy EE path")
policy_point, = ax.plot([], [], [], "o", label="Policy current")

start_point, = ax.plot([ee_pos0[0]], [ee_pos0[1]], [ee_pos0[2]],
                       "go", markersize=6, label="Start")
target_point, = ax.plot([tgt_pos[0]], [tgt_pos[1]], [tgt_pos[2]],
                        "r*", markersize=10, label="Target")

all_x = np.concatenate([demo_x, traj_x, [tgt_pos[0]]])
all_y = np.concatenate([demo_y, traj_y, [tgt_pos[1]]])
all_z = np.concatenate([demo_z, traj_z, [tgt_pos[2]]])

pad = 0.02  # 2cm padding
ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
ax.set_zlim(all_z.min()-pad, all_z.max()+pad)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Demo vs BC Policy Trajectory (both animated)")
ax.legend()

num_frames = max(len(demo_x), len(traj_x))

def init():
    demo_line.set_data([], [])
    demo_line.set_3d_properties([])
    demo_point.set_data([], [])
    demo_point.set_3d_properties([])

    policy_line.set_data([], [])
    policy_line.set_3d_properties([])
    policy_point.set_data([], [])
    policy_point.set_3d_properties([])

    return (demo_line, demo_point, policy_line, policy_point,
            start_point, target_point)

def update(i):
    di = min(i, len(demo_x) - 1)
    demo_line.set_data(demo_x[:di+1], demo_y[:di+1])
    demo_line.set_3d_properties(demo_z[:di+1])
    demo_point.set_data([demo_x[di]], [demo_y[di]])
    demo_point.set_3d_properties([demo_z[di]])

    pi = min(i, len(traj_x) - 1)
    policy_line.set_data(traj_x[:pi+1], traj_y[:pi+1])
    policy_line.set_3d_properties(traj_z[:pi+1])
    policy_point.set_data([traj_x[pi]], [traj_y[pi]])
    policy_point.set_3d_properties([traj_z[pi]])

    return (demo_line, demo_point, policy_line, policy_point,
            start_point, target_point)

ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    init_func=init,
    interval=20,
    blit=False
)

plt.show()
