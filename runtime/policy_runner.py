#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""IOA runtime utilities: checkpointed EE delta policy runner (robot-agnostic).

ROS2, MoveIt, and robot-specific layers are intentionally excluded; this module:

- loads a trained EE-delta BC policy (PyTorch checkpoint)
- computes error vectors err_vec = [pos_err(3), rpy_err(3)] from EE vs target poses
- feeds the error into the policy to predict ΔEE = [Δpos(3), Δrpy(3)]
- optionally converts ΔEE + dt into EE velocities (v_pos, v_rot)

Per-robot IK/servo control and joint command publication are handled elsewhere.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class EEDeltaPolicy(nn.Module):
    """Simple MLP policy: obs(6) -> act(6) = [d_pos(3), d_rpy(3)]."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def wrap_to_pi(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert quaternion (x, y, z, w) to roll, pitch, yaw in radians."""
    # roll (x-axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class IOAPolicyRunner:
    """Robot-agnostic EE-delta policy runner.

    Assumes the checkpoint contains:
        - "model_state_dict"
        - "obs_mean", "obs_std"
        - "act_mean", "act_std"

    Observations are err_vec = [pos_err(3), rpy_err(3)] computed in the base frame.
    """

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = Path(ckpt_path)

        # Model and normalization stats
        self.model = EEDeltaPolicy(obs_dim=6, act_dim=6, hidden_dim=128).to(self.device)
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None

        self._load_policy(self.ckpt_path)

    # ------------------------------------------------------------------
    # Checkpoint loading / normalization restoration
    # ------------------------------------------------------------------
    def _load_policy(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Restore normalization stats saved during training.
        self.obs_mean = np.array(ckpt["obs_mean"]).reshape(1, -1).astype(np.float32)
        self.obs_std = np.array(ckpt["obs_std"]).reshape(1, -1).astype(np.float32)
        self.act_mean = np.array(ckpt["act_mean"]).reshape(1, -1).astype(np.float32)
        self.act_std = np.array(ckpt["act_std"]).reshape(1, -1).astype(np.float32)

    # ------------------------------------------------------------------
    # Error vector utility: EE pose vs target pose
    # ------------------------------------------------------------------
    @staticmethod
    def compute_error_from_poses(
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        """Compute error vector [pos_err(3), rpy_err(3)] in base frame.

        Args:
            ee_pos: np.array shape (3,), EE position (x, y, z).
            ee_quat: np.array shape (4,), EE quaternion (x, y, z, w).
            target_pos: np.array shape (3,), target position.
            target_quat: np.array shape (4,), target quaternion.

        Returns:
            err_vec: np.array shape (6,), [pos_err, rpy_err] where
                     pos_err = target_pos - ee_pos,
                     rpy_err = wrap_to_pi(target_rpy - ee_rpy).
        """
        ee_pos = np.asarray(ee_pos, dtype=np.float32).reshape(3)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        ee_quat = np.asarray(ee_quat, dtype=np.float32).reshape(4)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)

        pos_err = target_pos - ee_pos

        ee_rpy = np.array(
            quat_to_rpy(ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]),
            dtype=np.float32,
        )
        tgt_rpy = np.array(
            quat_to_rpy(target_quat[0], target_quat[1], target_quat[2], target_quat[3]),
            dtype=np.float32,
        )

        rpy_err = tgt_rpy - ee_rpy
        rpy_err = np.array([wrap_to_pi(a) for a in rpy_err], dtype=np.float32)

        err_vec = np.concatenate([pos_err, rpy_err], axis=0)
        return err_vec

    # ------------------------------------------------------------------
    # Policy invocation: err_vec -> d_ee
    # ------------------------------------------------------------------
    def predict_delta(self, err_vec: np.ndarray) -> np.ndarray:
        """Predict ΔEE = [d_pos(3), d_rpy(3)] from error vector.

        Args:
            err_vec: np.array shape (6,), [pos_err(3), rpy_err(3)].

        Returns:
            d_ee: np.array shape (6,), [d_pos(3), d_rpy(3)].
        """
        if self.obs_mean is None or self.obs_std is None:
            raise RuntimeError("Observation normalization stats are not loaded.")
        if self.act_mean is None or self.act_std is None:
            raise RuntimeError("Action normalization stats are not loaded.")

        err_vec = np.asarray(err_vec, dtype=np.float32).reshape(1, -1)
        obs_norm = (err_vec - self.obs_mean) / (self.obs_std + 1e-8)
        obs_tensor = torch.from_numpy(obs_norm).float().to(self.device)

        with torch.no_grad():
            act_norm = self.model(obs_tensor).cpu().numpy()[0]

        d_ee = act_norm * (self.act_std.squeeze() + 1e-8) + self.act_mean.squeeze()
        return d_ee.astype(np.float32)

    def __call__(self, err_vec: np.ndarray) -> np.ndarray:
        """Alias for predict_delta(err_vec)."""
        return self.predict_delta(err_vec)

    # ------------------------------------------------------------------
    # Optional ΔEE + dt -> velocity conversion
    # ------------------------------------------------------------------
    @staticmethod
    def delta_to_twist(
        d_ee: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ΔEE into linear/angular velocities given dt.

        Args:
            d_ee: np.array shape (6,), [d_pos(3), d_rpy(3)].
            dt: control period in seconds.

        Returns:
            v_pos: np.array shape (3,), linear velocity.
            v_rot: np.array shape (3,), angular velocity (rpy rates).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        d_ee = np.asarray(d_ee, dtype=np.float32).reshape(-1)
        d_pos = d_ee[0:3]
        d_rpy = d_ee[3:6]

        v_pos = d_pos / dt
        v_rot = d_rpy / dt
        return v_pos.astype(np.float32), v_rot.astype(np.float32)


if __name__ == "__main__":
    # Simple usage example (dummy values):
    ckpt = "path/to/ee_bc_policy_1.pt"
    runner = IOAPolicyRunner(ckpt)

    # Dummy EE / target poses (same pose -> zero error)
    ee_pos = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    tgt_pos = np.array([0.3, 0.0, 0.2], dtype=np.float32)
    tgt_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    err = IOAPolicyRunner.compute_error_from_poses(ee_pos, ee_quat, tgt_pos, tgt_quat)
    d_ee = runner.predict_delta(err)
    v_pos, v_rot = IOAPolicyRunner.delta_to_twist(d_ee, dt=1.0 / 60.0)

    print("err_vec:", err)
    print("d_ee:", d_ee)
    print("v_pos:", v_pos)
    print("v_rot:", v_rot)
