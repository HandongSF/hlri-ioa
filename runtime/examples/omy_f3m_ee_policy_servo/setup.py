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

from setuptools import setup

package_name = 'omy_f3m_ee_policy_servo'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/ee_policy_servo.launch.py',
            'launch/omy_f3m_all.launch.py',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'torch'],
    zip_safe=True,
    maintainer='kiro',
    maintainer_email='kiro@example.com',
    description='EE BC policy -> MoveIt Servo + JointState bridge for OMY-F3M',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ee_policy_servo = omy_f3m_ee_policy_servo.ee_policy_servo_node:main',
            'ee_swing_node = omy_f3m_ee_policy_servo.ee_swing_node:main',
        ],
    },
)
