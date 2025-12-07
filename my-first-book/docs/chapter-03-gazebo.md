---
sidebar_position: 4
---

# Chapter 3: Gazebo Simulation

## Overview

Gazebo is an open-source 3D robotics simulator that allows you to test algorithms, design robots, and train AI systems in realistic scenarios without risking expensive hardware. It's the most widely used simulator in the ROS ecosystem.

## 3.1 Why Simulation?

### Benefits of Simulation

:::tip Beginner Explanation
Testing robots in simulation is like using a flight simulator before flying a real airplane. You can make mistakes, learn, and practice in a safe, controlled environment without any risk.
:::

**1. Safety**
- No risk of damaging expensive hardware
- Test dangerous scenarios safely
- Develop and debug without physical constraints

**2. Speed and Efficiency**
- Faster than real-time possible
- Parallel testing of multiple scenarios
- Quick iteration during development

**3. Cost-Effective**
- No hardware wear and tear
- Test before building physical prototype
- Access to virtual environments and sensors

**4. Reproducibility**
- Exact replay of scenarios
- Controlled testing conditions
- Consistent benchmarking

**5. Scalability**
- Test with hundreds of robots
- Complex environments without physical space
- Various weather and lighting conditions

### Limitations of Simulation

:::info Advanced Note
The "reality gap" or "sim-to-real gap" refers to differences between simulated and real-world physics, sensors, and dynamics. Modern techniques like domain randomization and system identification help bridge this gap.
:::

**Challenges:**
1. **Physics approximations**: Simplified contact models
2. **Sensor noise**: Perfect vs. real noisy sensors
3. **Actuation**: Ideal motors vs. real dynamics
4. **Environmental factors**: Wind, temperature, wear
5. **Computation limits**: Real-time constraints

## 3.2 Gazebo Architecture

### Core Components

**1. Physics Engine**
- ODE (Open Dynamics Engine) - default
- Bullet - faster, good for collisions
- Simbody - accurate biomechanics
- DART - dynamic animation and robotics

**2. Rendering Engine**
- OGRE (Object-Oriented Graphics Rendering Engine)
- GPU-accelerated 3D rendering
- Shadows, lighting, materials

**3. Sensor Simulation**
- Cameras (RGB, depth)
- LIDAR/laser scanners
- IMU (Inertial Measurement Unit)
- Contact sensors
- GPS

**4. Plugin System**
- Extend functionality
- Custom physics, sensors, controllers
- Integration with ROS 2

### Gazebo Classic vs. Gazebo (Ignition/Gz)

**Gazebo Classic (versions 1-11)**
- Original implementation
- Widely used with ROS 1 and ROS 2
- Active but in maintenance mode

**Gazebo (formerly Ignition)**
- Complete rewrite
- Modular architecture
- Better performance and graphics
- Recommended for new projects with ROS 2

## 3.3 Installing Gazebo

### Installation with ROS 2

```bash
# Install Gazebo Classic (for ROS 2 Humble)
sudo apt install ros-humble-gazebo-ros-pkgs

# Or install Gazebo (Ignition) - latest
sudo apt install ros-humble-ros-gz

# Verify installation
gazebo --version
```

### Testing Installation

```bash
# Launch Gazebo with ROS 2
gazebo

# Or with Gazebo (Ignition)
gz sim
```

## 3.4 URDF and Robot Description

### What is URDF?

**URDF (Unified Robot Description Format)** is an XML format for describing robot kinematics, dynamics, and visual properties.

:::tip Beginner Explanation
URDF is like a blueprint for your robot. It describes what parts the robot has (links), how they're connected (joints), what they look like, and how they move.
:::

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0"
               iyy="0.5" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0"
               iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Neck joint connecting head to base -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
  </joint>

</robot>
```

### Joint Types

1. **Fixed**: No movement (e.g., camera mount)
2. **Revolute**: Rotational with limits (e.g., elbow, knee)
3. **Continuous**: Continuous rotation (e.g., wheel)
4. **Prismatic**: Linear sliding (e.g., elevator, gripper)
5. **Floating**: 6-DOF (rarely used)
6. **Planar**: 2D movement

### XACRO - Programmable URDF

XACRO (XML Macros) extends URDF with programming features:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">

  <!-- Properties (variables) -->
  <xacro:property name="leg_length" value="0.8"/>
  <xacro:property name="leg_radius" value="0.05"/>
  <xacro:property name="leg_mass" value="3.0"/>

  <!-- Macros (functions) -->
  <xacro:macro name="leg" params="prefix reflect">
    <link name="${prefix}_upper_leg">
      <visual>
        <geometry>
          <cylinder length="${leg_length}" radius="${leg_radius}"/>
        </geometry>
      </visual>
      <inertial>
        <mass value="${leg_mass}"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${prefix}_hip" type="revolute">
      <parent link="base_link"/>
      <child link="${prefix}_upper_leg"/>
      <origin xyz="0 ${reflect * 0.1} -0.1" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:leg prefix="left" reflect="1"/>
  <xacro:leg prefix="right" reflect="-1"/>

</robot>
```

```bash
# Convert XACRO to URDF
xacro robot.urdf.xacro > robot.urdf
```

## 3.5 Gazebo Plugins

Gazebo plugins extend functionality and integrate with ROS 2.

### Differential Drive Plugin

```xml
<gazebo>
  <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <update_rate>50</update_rate>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <command_topic>cmd_vel</command_topic>
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>false</publish_wheel_tf>
  </plugin>
</gazebo>
```

### Camera Plugin

```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="front_camera">
    <update_rate>30.0</update_rate>
    <camera name="front">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>1920</width>
        <height>1080</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>front_camera</cameraName>
      <imageTopicName>/front_camera/image_raw</imageTopicName>
      <cameraInfoTopicName>/front_camera/camera_info</cameraInfoTopicName>
      <frameName>camera_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Plugin

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
      <ros>
        <namespace>/</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

### Joint State Publisher

```xml
<gazebo>
  <plugin name="gazebo_ros_joint_state_publisher"
          filename="libgazebo_ros_joint_state_publisher.so">
    <update_rate>50</update_rate>
    <joint_name>neck_joint</joint_name>
    <joint_name>left_hip_joint</joint_name>
    <joint_name>right_hip_joint</joint_name>
    <!-- Add all joints -->
  </plugin>
</gazebo>
```

## 3.6 Launching Robot in Gazebo

### Launch File Structure

```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    pkg_humanoid_description = get_package_share_directory('humanoid_description')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Path to URDF/XACRO file
    urdf_file = os.path.join(pkg_humanoid_description, 'urdf', 'humanoid.urdf.xacro')

    # Process xacro file
    robot_description = {'robot_description': Command(['xacro ', urdf_file])}

    # Start Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ]),
        launch_arguments={'world': 'empty.world'}.items()
    )

    # Start Gazebo client
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ])
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'humanoid_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    return LaunchDescription([
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        spawn_entity
    ])
```

```bash
# Launch the simulation
ros2 launch humanoid_gazebo humanoid_world.launch.py
```

## 3.7 Creating Custom Worlds

### World File Format (.world)

```xml
<?xml version="1.0"?>
<sdf version="1.6">
  <world name="humanoid_world">

    <!-- Physics settings -->
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Custom obstacle -->
    <model name="obstacle">
      <static>true</static>
      <pose>2 0 0.5 0 0 0</pose>
      <link name="box">
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 2.0 1.0</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 2.0 1.0</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Stairs model -->
    <include>
      <uri>model://cafe_table</uri>
      <pose>-2 2 0 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### Loading Custom World

```python
# In launch file
gazebo_server = IncludeLaunchDescription(
    PythonLaunchDescriptionSource([...]),
    launch_arguments={
        'world': os.path.join(pkg_dir, 'worlds', 'humanoid_world.world')
    }.items()
)
```

## 3.8 Controlling the Robot in Gazebo

### Joint Position Control

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Publisher to joint command topic
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.05, self.control_callback)
        self.time = 0.0

    def control_callback(self):
        msg = Float64MultiArray()

        # Sinusoidal joint motion for testing
        self.time += 0.05
        angle = math.sin(self.time)

        # Command for multiple joints
        msg.data = [
            angle,          # neck
            angle * 0.5,    # left_shoulder
            -angle * 0.5,   # right_shoulder
            angle * 0.3,    # left_hip
            -angle * 0.3    # right_hip
        ]

        self.pub.publish(msg)

def main():
    rclpy.init()
    controller = JointController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

### Velocity Control

```python
from geometry_msgs.msg import Twist

class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_velocity)

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.5   # Forward velocity (m/s)
        msg.angular.z = 0.2  # Angular velocity (rad/s)
        self.pub.publish(msg)
```

## 3.9 Using ros2_control with Gazebo

`ros2_control` is a framework for real-time control of robots, working both in simulation and on real hardware.

### URDF Configuration

```xml
<ros2_control name="humanoid_system" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>

  <!-- Joint interfaces -->
  <joint name="neck_joint">
    <command_interface name="position">
      <param name="min">-1.57</param>
      <param name="max">1.57</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>

  <!-- Repeat for all joints -->
  <joint name="left_hip_joint">
    <command_interface name="position"/>
    <command_interface name="velocity"/>
    <command_interface name="effort"/>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>

</ros2_control>
```

### Controller Configuration (YAML)

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    position_controller:
      type: position_controllers/JointGroupPositionController

    effort_controller:
      type: effort_controllers/JointGroupEffortController

position_controller:
  ros__parameters:
    joints:
      - neck_joint
      - left_shoulder_joint
      - right_shoulder_joint
      - left_hip_joint
      - right_hip_joint
      - left_knee_joint
      - right_knee_joint

effort_controller:
  ros__parameters:
    joints:
      - left_ankle_joint
      - right_ankle_joint
```

### Loading Controllers

```bash
# List controllers
ros2 control list_controllers

# Load and start controller
ros2 control load_controller position_controller
ros2 control set_controller_state position_controller start

# Switch controllers
ros2 control switch_controllers \
  --activate position_controller \
  --deactivate effort_controller
```

## 3.10 Practical Example: Humanoid Walking in Gazebo

### Complete Walking Controller

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
import numpy as np

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Walking parameters
        self.step_length = 0.3   # meters
        self.step_height = 0.05  # meters
        self.step_period = 1.0   # seconds per step
        self.time = 0.0

        # Joint names
        self.joint_names = [
            'left_hip_pitch', 'left_knee', 'left_ankle',
            'right_hip_pitch', 'right_knee', 'right_ankle'
        ]

        # Control loop
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Walking controller started')

    def joint_state_callback(self, msg):
        # Process current joint states if needed
        pass

    def control_loop(self):
        self.time += 0.01
        phase = (self.time % self.step_period) / self.step_period

        # Simple sinusoidal gait pattern
        # Left leg (stance when phase < 0.5, swing when phase >= 0.5)
        if phase < 0.5:
            # Left leg stance
            left_hip = -0.3 * (phase / 0.5)
            left_knee = 0.1
            left_ankle = 0.1 * (phase / 0.5)
        else:
            # Left leg swing
            swing_phase = (phase - 0.5) / 0.5
            left_hip = -0.3 + 0.6 * swing_phase
            left_knee = 0.5 * np.sin(swing_phase * np.pi)
            left_ankle = -0.1 + 0.2 * swing_phase

        # Right leg (opposite phase)
        if phase < 0.5:
            # Right leg swing
            swing_phase = phase / 0.5
            right_hip = 0.3 - 0.6 * swing_phase
            right_knee = 0.5 * np.sin(swing_phase * np.pi)
            right_ankle = 0.1 - 0.2 * swing_phase
        else:
            # Right leg stance
            stance_phase = (phase - 0.5) / 0.5
            right_hip = -0.3 * stance_phase
            right_knee = 0.1
            right_ankle = 0.1 * stance_phase

        # Publish commands
        cmd = Float64MultiArray()
        cmd.data = [
            left_hip, left_knee, left_ankle,
            right_hip, right_knee, right_ankle
        ]
        self.joint_cmd_pub.publish(cmd)

def main():
    rclpy.init()
    controller = WalkingController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

1. Gazebo provides a powerful, realistic simulation environment for robots
2. URDF/XACRO describes robot structure, kinematics, and dynamics
3. Plugins extend Gazebo functionality and integrate with ROS 2
4. Custom worlds allow testing in diverse environments
5. ros2_control provides unified interface for sim and real hardware
6. Simulation enables safe, fast, and cost-effective development
7. Understanding the sim-to-real gap is crucial for real-world deployment

## Exercises

### Beginner
1. Install Gazebo and launch the default empty world
2. Create a simple URDF for a 2-link robot arm
3. Spawn your robot in Gazebo and visualize in RViz
4. Add a camera sensor to your robot

### Intermediate
1. Create a custom world with obstacles and terrain
2. Implement a XACRO macro for a humanoid leg
3. Add IMU and LIDAR sensors with appropriate plugins
4. Control your robot using ros2_control

### Advanced
1. Implement a complete bipedal walking gait
2. Create a realistic humanoid model with proper mass/inertia
3. Integrate depth cameras and implement SLAM
4. Develop sim-to-real transfer techniques

## Additional Resources

- **Gazebo Tutorials**: [https://gazebosim.org/docs](https://gazebosim.org/docs)
- **URDF Tutorials**: ROS 2 official documentation
- **ros2_control**: [https://control.ros.org](https://control.ros.org)
- **Examples**: gazebo_ros_demos package
- **Community**: Gazebo Community Forum

---

**Next Chapter**: [NVIDIA Isaac Sim](/docs/chapter-04-isaac)
