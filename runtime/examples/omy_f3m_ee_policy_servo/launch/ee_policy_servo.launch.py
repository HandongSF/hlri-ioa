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
from launch_ros.actions import Node

def generate_launch_description():
    ee_policy_node = Node(
        package='omy_f3m_ee_policy_servo',
        executable='ee_policy_servo',
        name='ee_policy_servo',
        output='screen',
        parameters=[
            {
                'policy_path': 'install/omy_f3m_ee_policy_servo/share/omy_f3m_ee_policy_servo/ee_bc_policy_1.pt',
                'base_frame': 'link0',
                'ee_frame': 'end_effector_link',
                'target_frame': 'target_cube',
                'servo_twist_topic': '/servo_node/delta_twist_cmds',
                'rate_hz': 60.0,
            }
        ],
    )

    return LaunchDescription([ee_policy_node])
