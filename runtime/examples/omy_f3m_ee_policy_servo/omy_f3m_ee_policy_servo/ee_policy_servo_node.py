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

import math
import numpy as np
import torch
import torch.nn as nn
import os
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped

class EEDeltaPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def wrap_to_pi(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_rpy(qx, qy, qz, qw):
    """
    Convert quaternion (x, y, z, w) to roll, pitch, yaw in radians.
    """
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

class EEPolicyToServoNode(Node):
    """
    Reads EE and target poses from TF, feeds the error to the learned BC policy,
    and publishes the resulting twist in base_frame to MoveIt Servo.
    """

    def __init__(self):
        super().__init__("ee_policy_servo")

        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.parameter.Parameter.Type.BOOL,
                True
            )
        ])

        self.declare_parameter(
            "policy_path",
            "install/omy_f3m_ee_policy_servo/share/omy_f3m_ee_policy_servo/ee_bc_policy_1.pt",
        )
        self.declare_parameter("base_frame", "link0")
        self.declare_parameter("ee_frame", "end_effector_link")
        self.declare_parameter("target_frame", "target_cube")
        self.declare_parameter("servo_twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("rate_hz", 60.0)

        policy_path = self.get_parameter("policy_path").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.ee_frame = self.get_parameter("ee_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.servo_twist_topic = self.get_parameter("servo_twist_topic").get_parameter_value().string_value
        self.rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value

        self.twist_pub = self.create_publisher(TwistStamped, self.servo_twist_topic, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        self.model = EEDeltaPolicy(obs_dim=6, act_dim=6, hidden_dim=128).to(self.device)
        self.obs_mean = None
        self.obs_std = None
        self.act_mean = None
        self.act_std = None

        self._load_policy(policy_path)

        self.dt = 1.0 / self.rate_hz
        self.timer = self.create_timer(self.dt, self.timer_callback)

        use_sim_time_val = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.get_logger().info(
            f"EEPolicyToServoNode started. rate={self.rate_hz}Hz, "
            f"base={self.base_frame}, ee={self.ee_frame}, target={self.target_frame}, "
            f"use_sim_time={use_sim_time_val}"
        )

    def _load_policy(self, path: str):
        if not os.path.isfile(path):
            self.get_logger().error(f"Policy checkpoint not found: {path}")
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Restore normalization stats saved during training.
        self.obs_mean = np.array(ckpt["obs_mean"]).reshape(1, -1).astype(np.float32)
        self.obs_std = np.array(ckpt["obs_std"]).reshape(1, -1).astype(np.float32)
        self.act_mean = np.array(ckpt["act_mean"]).reshape(1, -1).astype(np.float32)
        self.act_std = np.array(ckpt["act_std"]).reshape(1, -1).astype(np.float32)

        self.get_logger().info(f"Loaded policy from {path}")

    def _lookup_transform(self, target_frame: str, source_frame: str) -> TransformStamped | None:
        """Lookup source_frame pose in target_frame."""
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.01),
            )
            return t
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup failed {target_frame} <- {source_frame}: {str(ex)}"
            )
            return None

    def timer_callback(self):
        t_ee = self._lookup_transform(self.base_frame, self.ee_frame)
        t_target = self._lookup_transform(self.base_frame, self.target_frame)

        if t_ee is None or t_target is None:
            # Publish zero twist when TF lookup fails.
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.base_frame
            self.twist_pub.publish(twist)
            return

        ee_pos = np.array(
            [
                t_ee.transform.translation.x,
                t_ee.transform.translation.y,
                t_ee.transform.translation.z,
            ],
            dtype=np.float32,
        )

        target_pos = np.array(
            [
                t_target.transform.translation.x,
                t_target.transform.translation.y,
                t_target.transform.translation.z,
            ],
            dtype=np.float32,
        )

        ee_q = t_ee.transform.rotation
        tgt_q = t_target.transform.rotation

        ee_rpy = np.array(
            quat_to_rpy(ee_q.x, ee_q.y, ee_q.z, ee_q.w),
            dtype=np.float32,
        )
        tgt_rpy = np.array(
            quat_to_rpy(tgt_q.x, tgt_q.y, tgt_q.z, tgt_q.w),
            dtype=np.float32,
        )

        pos_err = target_pos - ee_pos
        rpy_err = tgt_rpy - ee_rpy
        rpy_err = np.array([wrap_to_pi(a) for a in rpy_err], dtype=np.float32)

        err_vec = np.concatenate([pos_err, rpy_err], axis=0)

        obs_norm = (err_vec - self.obs_mean.squeeze()) / (self.obs_std.squeeze() + 1e-8)
        obs_tensor = (
            torch.from_numpy(obs_norm).float().to(self.device).unsqueeze(0)
        )

        with torch.no_grad():
            act_norm = self.model(obs_tensor)
            act_norm = act_norm.cpu().numpy()[0]

        d_ee = act_norm * (self.act_std.squeeze() + 1e-8) + self.act_mean.squeeze()
        d_pos = d_ee[0:3]
        d_rpy = d_ee[3:6]

        v_pos = d_pos / self.dt
        v_rot = d_rpy / self.dt

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        # Publish twist with base_frame as planning frame.
        twist_msg.header.frame_id = self.base_frame

        twist_msg.twist.linear.x = float(v_pos[0])
        twist_msg.twist.linear.y = float(v_pos[1])
        twist_msg.twist.linear.z = float(v_pos[2])
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = 0.0

        self.twist_pub.publish(twist_msg)

class TrajectoryToJointCommandBridge(Node):
    """Bridge MoveIt Servo JointTrajectory output to /joint_command JointState."""

    def __init__(self):
        super().__init__("traj_to_joint_command_bridge")

        self.declare_parameter("servo_traj_topic", "/servo_node/joint_trajectory")
        self.declare_parameter("joint_command_topic", "/joint_command")

        self.servo_traj_topic = (
            self.get_parameter("servo_traj_topic").get_parameter_value().string_value
        )
        self.joint_command_topic = (
            self.get_parameter("joint_command_topic")
            .get_parameter_value()
            .string_value
        )

        self.traj_sub = self.create_subscription(
            JointTrajectory,
            self.servo_traj_topic,
            self.traj_callback,
            10,
        )

        self.joint_pub = self.create_publisher(
            JointState, self.joint_command_topic, 10
        )

        self.get_logger().info(
            f"TrajectoryToJointCommandBridge started. "
            f"Sub: {self.servo_traj_topic} (JointTrajectory) "
            f"-> Pub: {self.joint_command_topic} (JointState)"
        )

    def traj_callback(self, msg: JointTrajectory):
        if not msg.points:
            return

        # MoveIt Servo typically streams single-point trajectories; use the last point for safety.
        pt = msg.points[-1]

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(msg.joint_names)
        js.position = list(pt.positions)
        js.velocity = list(pt.velocities)
        js.effort = list(pt.effort)

        self.joint_pub.publish(js)

def main(args=None):
    rclpy.init(args=args)

    ee_policy_node = EEPolicyToServoNode()
    bridge_node = TrajectoryToJointCommandBridge()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ee_policy_node)
    executor.add_node(bridge_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ee_policy_node.destroy_node()
        bridge_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"[ee_policy_servo] Ignore shutdown error: {e}")


if __name__ == "__main__":
    main()
