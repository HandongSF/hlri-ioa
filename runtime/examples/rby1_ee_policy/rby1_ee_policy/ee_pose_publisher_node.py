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
"""ROS 2 node that publishes the RBY1 EE pose, error, and BC policy outputs."""

from __future__ import annotations

from typing import Optional, Tuple
import math
import os

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import torch
import torch.nn as nn

from geometry_msgs.msg import PoseStamped

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import TransformStamped


def quaternion_multiply(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Multiply two quaternions represented as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert quaternion (x, y, z, w) to roll, pitch, yaw."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert roll, pitch, yaw to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


class EEDeltaPolicy(nn.Module):
    """Simple feedforward network used to decode the policy checkpoint."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 6, hidden_dim: int = 128) -> None:
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


class RBY1EEPosePublisher(Node):
    """Publish RBY1 EE pose, error, and BC-predicted EE delta/next pose in the base frame."""

    def __init__(self) -> None:
        super().__init__('rby1_ee_pose_publisher')

        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('ee_frame', 'right_tcp')
        self.declare_parameter('target_topic', '/dish_target')
        self.declare_parameter('ee_pose_topic', '/rby1/ee_pose')
        self.declare_parameter('error_topic', '/rby1/ee_to_dish_error')
        self.declare_parameter('rate_hz', 60.0)
        self.declare_parameter(
            'policy_path',
            'install/omy_f3m_ee_policy_servo/share/omy_f3m_ee_policy_servo/ee_bc_policy_1.pt',
        )
        self.declare_parameter('policy_pose_topic', '/rby1/ee_policy_pose')
        self.declare_parameter('policy_delta_topic', '/rby1/ee_policy_delta')

        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.ee_frame = self.get_parameter('ee_frame').get_parameter_value().string_value
        self.target_topic = self.get_parameter('target_topic').get_parameter_value().string_value
        self.ee_pose_topic = self.get_parameter('ee_pose_topic').get_parameter_value().string_value
        self.error_topic = self.get_parameter('error_topic').get_parameter_value().string_value
        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.policy_pose_topic = self.get_parameter('policy_pose_topic').get_parameter_value().string_value
        self.policy_delta_topic = self.get_parameter('policy_delta_topic').get_parameter_value().string_value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.target_pose: Optional[PoseStamped] = None
        self.target_sub = self.create_subscription(
            PoseStamped,
            self.target_topic,
            self._target_callback,
            10,
        )

        self.ee_pose_pub = self.create_publisher(PoseStamped, self.ee_pose_topic, 10)
        self.error_pub = self.create_publisher(PoseStamped, self.error_topic, 10)
        self.policy_pose_pub = self.create_publisher(PoseStamped, self.policy_pose_topic, 10)
        self.policy_delta_pub = self.create_publisher(PoseStamped, self.policy_delta_topic, 10)

        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0), self._timer_callback)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Loading policy on device '{self.device}' from {self.policy_path}")
        self.model = EEDeltaPolicy(obs_dim=6, act_dim=6, hidden_dim=128).to(self.device)
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        self.act_mean: Optional[np.ndarray] = None
        self.act_std: Optional[np.ndarray] = None
        self._load_policy(self.policy_path)

        self.get_logger().info(
            f"RBY1 EE pose publisher ready (base='{self.base_frame}', ee='{self.ee_frame}', "
            f"target_topic='{self.target_topic}')."
        )

    def _target_callback(self, msg: PoseStamped) -> None:
        self.target_pose = msg

    def _lookup_transform(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed for {target_frame} <- {source_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _load_policy(self, path: str) -> None:
        if not os.path.isfile(path):
            self.get_logger().error(f"Policy checkpoint not found at '{path}'.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to load policy checkpoint: {exc}")
            return

        state_dict = checkpoint.get('model_state_dict')
        if state_dict is None:
            self.get_logger().error('Checkpoint missing model_state_dict entry.')
            return

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.obs_mean = np.array(checkpoint.get('obs_mean', []), dtype=np.float32).reshape(1, -1)
        self.obs_std = np.array(checkpoint.get('obs_std', []), dtype=np.float32).reshape(1, -1)
        self.act_mean = np.array(checkpoint.get('act_mean', []), dtype=np.float32).reshape(1, -1)
        self.act_std = np.array(checkpoint.get('act_std', []), dtype=np.float32).reshape(1, -1)

        self.get_logger().info('Successfully loaded ee_bc_policy_1 checkpoint.')

    def _timer_callback(self) -> None:
        transform = self._lookup_transform(self.base_frame, self.ee_frame)
        if transform is None:
            return

        ee_pose = PoseStamped()
        ee_pose.header.stamp = self.get_clock().now().to_msg()
        ee_pose.header.frame_id = self.base_frame
        ee_pose.pose.position.x = transform.transform.translation.x
        ee_pose.pose.position.y = transform.transform.translation.y
        ee_pose.pose.position.z = transform.transform.translation.z
        ee_pose.pose.orientation = transform.transform.rotation
        self.ee_pose_pub.publish(ee_pose)

        if self.target_pose is None:
            return

        target_in_base = self._target_pose_in_base()
        if target_in_base is None:
            return

        error_pose = PoseStamped()
        error_pose.header = ee_pose.header

        target_vec = np.array(
            [
                target_in_base.pose.position.x,
                target_in_base.pose.position.y,
                target_in_base.pose.position.z,
            ],
            dtype=float,
        )
        ee_vec = np.array(
            [
                ee_pose.pose.position.x,
                ee_pose.pose.position.y,
                ee_pose.pose.position.z,
            ],
            dtype=float,
        )
        pos_err = target_vec - ee_vec
        error_pose.pose.position.x = float(pos_err[0])
        error_pose.pose.position.y = float(pos_err[1])
        error_pose.pose.position.z = float(pos_err[2])

        q_target = target_in_base.pose.orientation
        q_ee = ee_pose.pose.orientation

        tgt_rpy = quat_to_rpy(q_target.x, q_target.y, q_target.z, q_target.w)
        ee_rpy = quat_to_rpy(q_ee.x, q_ee.y, q_ee.z, q_ee.w)

        rpy_err = np.array(
            [
                wrap_to_pi(tgt_rpy[0] - ee_rpy[0]),
                wrap_to_pi(tgt_rpy[1] - ee_rpy[1]),
                wrap_to_pi(tgt_rpy[2] - ee_rpy[2]),
            ],
            dtype=np.float32,
        )

        q_err = rpy_to_quat(float(rpy_err[0]), float(rpy_err[1]), float(rpy_err[2]))
        error_pose.pose.orientation.x = q_err[0]
        error_pose.pose.orientation.y = q_err[1]
        error_pose.pose.orientation.z = q_err[2]
        error_pose.pose.orientation.w = q_err[3]

        self.error_pub.publish(error_pose)

        policy_delta = self._compute_policy_delta(pos_err=pos_err, rpy_err=rpy_err)
        if policy_delta is None:
            return

        d_pos, d_rpy = policy_delta

        delta_msg = PoseStamped()
        delta_msg.header = ee_pose.header
        delta_msg.pose.position.x = float(d_pos[0])
        delta_msg.pose.position.y = float(d_pos[1])
        delta_msg.pose.position.z = float(d_pos[2])
        q_delta = rpy_to_quat(float(d_rpy[0]), float(d_rpy[1]), float(d_rpy[2]))
        delta_msg.pose.orientation.x = q_delta[0]
        delta_msg.pose.orientation.y = q_delta[1]
        delta_msg.pose.orientation.z = q_delta[2]
        delta_msg.pose.orientation.w = q_delta[3]
        self.policy_delta_pub.publish(delta_msg)

        policy_target = PoseStamped()
        policy_target.header = ee_pose.header
        policy_target.pose.position.x = ee_pose.pose.position.x + float(d_pos[0])
        policy_target.pose.position.y = ee_pose.pose.position.y + float(d_pos[1])
        policy_target.pose.position.z = ee_pose.pose.position.z + float(d_pos[2])

        q_ee_tuple = (
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w,
        )
        q_next = quaternion_multiply(q_delta, q_ee_tuple)
        policy_target.pose.orientation.x = q_next[0]
        policy_target.pose.orientation.y = q_next[1]
        policy_target.pose.orientation.z = q_next[2]
        policy_target.pose.orientation.w = q_next[3]
        self.policy_pose_pub.publish(policy_target)

    def _target_pose_in_base(self) -> Optional[PoseStamped]:
        if self.target_pose is None:
            return None

        if self.target_pose.header.frame_id == self.base_frame:
            return self.target_pose

        try:
            tf_target = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_pose.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup for dish target failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        return do_transform_pose(self.target_pose, tf_target)

    def _policy_delta_stats_ready(self) -> bool:
        arrays = [self.obs_mean, self.obs_std, self.act_mean, self.act_std]
        return all(isinstance(arr, np.ndarray) and arr.size >= 6 for arr in arrays)

    def _compute_policy_delta(
        self,
        pos_err: np.ndarray,
        rpy_err: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run the BC policy to compute delta translation/rotation."""
        if not self._policy_delta_stats_ready():
            self.get_logger().warn(
                'Policy normalization statistics not ready; skip publishing BC command.',
                throttle_duration_sec=2.0,
            )
            return None

        err_vec = np.concatenate(
            [pos_err.astype(np.float32), rpy_err.astype(np.float32)],
            axis=0,
        )

        obs_mean = self.obs_mean.squeeze()
        obs_std = self.obs_std.squeeze()
        act_mean = self.act_mean.squeeze()
        act_std = self.act_std.squeeze()

        if err_vec.shape[0] != obs_mean.shape[0]:
            self.get_logger().error(
                f'Observation dimension mismatch: got {err_vec.shape[0]}, expected {obs_mean.shape[0]}'
            )
            return None

        obs_norm = (err_vec - obs_mean) / (obs_std + 1e-8)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            act_norm = self.model(obs_tensor).cpu().numpy()[0]

        d_ee = act_norm * (act_std + 1e-8) + act_mean
        d_pos = d_ee[0:3]
        d_rpy = np.array([wrap_to_pi(angle) for angle in d_ee[3:6]], dtype=np.float32)
        return d_pos, d_rpy


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = RBY1EEPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
