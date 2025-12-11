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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    prefix = LaunchConfiguration("prefix")
    use_gui = LaunchConfiguration("use_gui")
    use_sim_time = LaunchConfiguration("use_sim_time")

    desc_pkg_share = FindPackageShare("open_manipulator_description")
    moveit_pkg_share = FindPackageShare("open_manipulator_moveit_config")

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                desc_pkg_share,
                "launch",
                "omy_f3m.launch.py",
            ])
        ),
        launch_arguments={
            "prefix": prefix,
            "use_gui": use_gui,
            "use_sim_time": use_sim_time,
        }.items(),
    )

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                moveit_pkg_share,
                "launch",
                "omy_f3m_servo.launch.py",
            ])
        ),
        launch_arguments={
            # Servo launch expects the argument name "use_sim".
            "use_sim": use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="prefix of the joint and link names",
        ),
        DeclareLaunchArgument(
            "use_gui",
            default_value="false",
            description="Run joint state publisher gui node",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock if true",
        ),

        description_launch,
        servo_launch,
    ])
