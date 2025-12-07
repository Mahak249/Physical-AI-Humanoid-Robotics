---
sidebar_position: 3
---

# Chapter 2: ROS 2 Fundamentals

## Overview

Robot Operating System 2 (ROS 2) is the industry-standard middleware for robot development. It provides tools, libraries, and conventions that simplify the creation of complex robot behavior across diverse robotic platforms.

## 2.1 What is ROS 2?

### History and Evolution

:::tip Beginner Explanation
Think of ROS 2 as the "operating system" for robots, just like Windows or macOS for your computer. It helps different parts of your robot talk to each other and provides tools to make programming robots easier.
:::

**ROS 1 (2007-2020)**
- Started at Stanford, developed by Willow Garage
- Single-master architecture
- Built on TCP/IP
- Became de facto standard in research

**ROS 2 (2017-Present)**
- Complete redesign for production systems
- Distributed architecture (no single master)
- Based on DDS (Data Distribution Service)
- Real-time capable
- Multi-platform (Linux, Windows, macOS)
- Security and safety features

### Why ROS 2 for Humanoid Robotics?

1. **Modular Design**: Separate perception, planning, and control
2. **Real-Time Performance**: Critical for balance control
3. **Distributed Computing**: Run components on multiple computers
4. **Large Ecosystem**: Thousands of packages and tools
5. **Industry Adoption**: Used by Tesla, Boston Dynamics, NASA

## 2.2 Core Concepts

### Nodes

**Definition**: A node is a process that performs computation.

:::info Advanced Note
Nodes are separate OS processes that communicate via DDS. This provides process isolation and allows leveraging multi-core systems efficiently. Each node runs in its own address space.
:::

**Example Nodes in a Humanoid Robot:**
- Camera driver node
- Object detection node
- Motion planning node
- Joint controller node
- Sensor fusion node

```python
# Simple ROS 2 node in Python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from ROS 2!')

def main():
    rclpy.init()
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

**Definition**: Topics are named buses for streaming data between nodes using a publish-subscribe pattern.

**Key Characteristics:**
- **Many-to-many**: Multiple publishers, multiple subscribers
- **Anonymous**: Publishers don't know about subscribers
- **Continuous**: Data flows continuously
- **Best-effort or reliable**: Configurable QoS

```python
# Publisher example
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        self.timer = self.create_timer(1.0, self.publish_status)

    def publish_status(self):
        msg = String()
        msg.data = 'Robot is operational'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
```

```python
# Subscriber example
class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10
        )

    def status_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

**Common Topics in Humanoid Robots:**
- `/camera/image_raw` - Camera images
- `/joint_states` - Joint positions and velocities
- `/cmd_vel` - Velocity commands
- `/tf` - Coordinate transformations
- `/scan` - LIDAR scan data

### Services

**Definition**: Services provide request-response communication (RPC - Remote Procedure Call).

**When to use Services vs Topics:**
- **Topics**: Continuous data streams (sensor readings, commands)
- **Services**: Occasional requests with responses (configuration, queries)

```python
# Service server
from example_interfaces.srv import AddTwoInts

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response
```

```python
# Service client
class ClientNode(Node):
    def __init__(self):
        super().__init__('client_node')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.client.call_async(request)
        return future
```

### Actions

**Definition**: Actions are for long-running tasks that need feedback and can be preempted.

**Use Cases:**
- Navigation to a goal (can monitor progress and cancel)
- Grasping an object (feedback on gripper position)
- Executing a motion plan

**Action has three parts:**
1. **Goal**: What to do
2. **Feedback**: Progress updates
3. **Result**: Final outcome

```python
# Simplified action server
from action_tutorials_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        # Execute the action and send feedback
        feedback_msg = Fibonacci.Feedback()
        sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            sequence.append(sequence[i] + sequence[i-1])
            feedback_msg.sequence = sequence
            goal_handle.publish_feedback(feedback_msg)
            await asyncio.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        return result
```

### Parameters

**Definition**: Parameters are node configuration values that can be set and modified at runtime.

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with defaults
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('robot_name', 'humanoid_01')

        # Get parameter values
        max_vel = self.get_parameter('max_velocity').value
        robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(f'Robot: {robot_name}, Max vel: {max_vel}')
```

```bash
# Set parameters from command line
ros2 run my_package parameter_node --ros-args -p max_velocity:=2.0

# Set parameters from YAML file
ros2 run my_package parameter_node --ros-args --params-file config.yaml
```

## 2.3 ROS 2 Workspace Structure

### Typical Workspace Layout

```
humanoid_ws/
├── src/                           # Source code
│   ├── humanoid_description/      # URDF robot models
│   ├── humanoid_control/          # Control nodes
│   ├── humanoid_perception/       # Vision and sensors
│   └── humanoid_bringup/          # Launch files
├── build/                         # Build artifacts
├── install/                       # Installed packages
└── log/                          # Build and runtime logs
```

### Package Structure

```
humanoid_control/
├── package.xml                    # Package manifest
├── setup.py                       # Python setup
├── CMakeLists.txt                # Build configuration (C++)
├── humanoid_control/             # Python package
│   ├── __init__.py
│   ├── joint_controller.py
│   └── balance_controller.py
├── launch/                       # Launch files
│   └── control.launch.py
├── config/                       # Configuration files
│   └── controllers.yaml
└── resource/                     # Resources
```

### Creating a Workspace

```bash
# Create workspace
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws/src

# Clone packages
git clone https://github.com/example/humanoid_control.git

# Build workspace
cd ~/humanoid_ws
colcon build

# Source the workspace
source install/setup.bash
```

## 2.4 Essential ROS 2 Tools

### Command Line Tools

```bash
# List running nodes
ros2 node list

# Get node info
ros2 node info /joint_controller

# List topics
ros2 topic list

# View topic data
ros2 topic echo /joint_states

# Get topic info (type, publishers, subscribers)
ros2 topic info /camera/image_raw

# Publish to a topic
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"

# Call a service
ros2 service call /reset_robot std_srvs/srv/Trigger

# List parameters of a node
ros2 param list /controller_node

# Get parameter value
ros2 param get /controller_node max_velocity

# Set parameter value
ros2 param set /controller_node max_velocity 2.0

# Record topics to a bag file
ros2 bag record -a  # Record all topics
ros2 bag record /joint_states /camera/image_raw  # Record specific topics

# Play back recorded data
ros2 bag play my_recording.db3
```

### RViz2 - Visualization Tool

RViz2 is the ROS 2 visualization tool for displaying sensor data, robot models, and more.

**Common Visualizations:**
- 3D robot model from URDF
- Camera images
- Point clouds from LIDAR
- Coordinate frames (TF)
- Trajectories and paths
- Markers and shapes

```bash
# Launch RViz2
rviz2

# Launch with a config file
rviz2 -d config/robot_view.rviz
```

### RQt - Qt-based GUI Tools

```bash
# Launch rqt with plugin chooser
rqt

# Common plugins:
# - rqt_graph: Visualize node and topic graph
# - rqt_plot: Plot numeric data over time
# - rqt_reconfigure: Dynamic parameter tuning
# - rqt_console: View log messages
# - rqt_image_view: Display images

# Launch specific tool
rqt_graph  # Visualize the computation graph
```

## 2.5 Launch Files

Launch files allow starting multiple nodes with configuration in a single command.

### Python Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='humanoid_perception',
            executable='camera_node',
            name='front_camera',
            parameters=[{'fps': 30, 'resolution': '1920x1080'}]
        ),
        Node(
            package='humanoid_perception',
            executable='object_detector',
            name='detector',
            remappings=[
                ('image_raw', '/front_camera/image_raw')
            ]
        ),
        Node(
            package='humanoid_control',
            executable='balance_controller',
            name='balance',
            output='screen'
        ),
    ])
```

```bash
# Launch the system
ros2 launch humanoid_bringup robot.launch.py
```

### Advanced Launch Features

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare arguments
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='true',
        description='Use simulation instead of real hardware'
    )

    # Use argument values
    use_sim = LaunchConfiguration('use_sim')

    return LaunchDescription([
        use_sim_arg,

        # Conditional node (only if use_sim is true)
        Node(
            package='gazebo_ros',
            executable='spawn_entity',
            condition=IfCondition(use_sim),
            arguments=['-entity', 'humanoid', '-file', 'model.urdf']
        ),

        # Include another launch file
        IncludeLaunchDescription(
            'path/to/other.launch.py',
            launch_arguments={'namespace': 'robot1'}.items()
        ),
    ])
```

```bash
# Pass arguments to launch file
ros2 launch humanoid_bringup robot.launch.py use_sim:=false
```

## 2.6 Message Types and Custom Messages

### Standard Message Types

```python
# Commonly used messages
from std_msgs.msg import String, Int32, Float64, Bool
from geometry_msgs.msg import Pose, Twist, Point, Vector3
from sensor_msgs.msg import Image, LaserScan, JointState, Imu
from nav_msgs.msg import Odometry, Path
```

### Creating Custom Messages

**Step 1: Define message in package**

Create `humanoid_interfaces/msg/JointCommand.msg`:
```
# JointCommand.msg
string joint_name
float64 position
float64 velocity
float64 effort
```

**Step 2: Update package.xml**
```xml
<depend>rosidl_default_generators</depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

**Step 3: Update CMakeLists.txt**
```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointCommand.msg"
)
```

**Step 4: Build and use**
```bash
colcon build --packages-select humanoid_interfaces

# Use in Python
from humanoid_interfaces.msg import JointCommand
```

## 2.7 Coordinate Frames and TF2

### The TF2 System

TF2 (Transform Library 2) manages coordinate frames and transformations between them.

:::tip Beginner Explanation
Imagine you're giving directions. You might say "the ball is 2 meters in front of me" or "the ball is 5 meters north of the house". These use different reference points (you vs. the house). TF2 helps robots convert between different reference points automatically.
:::

**Common Frames in Humanoid Robots:**
- `world` or `map`: Fixed global frame
- `odom`: Odometry frame (local, drift-prone)
- `base_link`: Robot's base center
- `torso`: Upper body center
- `head`: Head/camera mount
- `left_hand`, `right_hand`: Hand frames
- `left_foot`, `right_foot`: Foot contact frames

### Broadcasting Transforms

```python
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class FrameBroadcaster(Node):
    def __init__(self):
        super().__init__('frame_broadcaster')
        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.01, self.broadcast_transform)

    def broadcast_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'

        # Set translation
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.5

        # Set rotation (quaternion)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.br.sendTransform(t)
```

### Looking Up Transforms

```python
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point

class FrameListener(Node):
    def __init__(self):
        super().__init__('frame_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def transform_point(self, point, from_frame, to_frame):
        try:
            # Wait for transform to be available
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time()
            )

            # Transform the point
            point_transformed = do_transform_point(point, transform)
            return point_transformed

        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            return None
```

## 2.8 Quality of Service (QoS)

QoS settings control the behavior of communication between publishers and subscribers.

### Key QoS Policies

1. **Reliability**
   - `RELIABLE`: Guaranteed delivery (TCP-like)
   - `BEST_EFFORT`: Fast, may drop messages (UDP-like)

2. **Durability**
   - `VOLATILE`: Only to current subscribers
   - `TRANSIENT_LOCAL`: Late-joiners get cached messages

3. **History**
   - `KEEP_LAST`: Keep last N messages
   - `KEEP_ALL`: Keep all messages (memory permitting)

4. **Deadline**: Maximum expected time between messages
5. **Lifespan**: How long messages are valid
6. **Liveliness**: Detect if publisher is alive

### Setting QoS

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create custom QoS profile
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Use in publisher/subscriber
self.publisher = self.create_publisher(
    String,
    'my_topic',
    qos_profile
)
```

**Common Presets:**
```python
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

# For sensor data (best effort, volatile)
self.sub = self.create_subscription(
    Image,
    '/camera/image',
    callback,
    qos_profile_sensor_data
)
```

## 2.9 Building a Humanoid Robot Controller

Let's put it all together with a simple balance controller example:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Vector3
import numpy as np

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_commands',
            10
        )

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Parameters
        self.declare_parameter('kp', 50.0)
        self.declare_parameter('kd', 5.0)
        self.declare_parameter('target_angle', 0.0)

        # State variables
        self.current_angle = 0.0
        self.current_angular_vel = 0.0
        self.joint_positions = []

        # Control loop timer (100 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Balance controller initialized')

    def imu_callback(self, msg):
        # Simple pitch estimation from IMU
        # (In practice, use proper orientation estimation)
        accel = msg.linear_acceleration
        self.current_angle = np.arctan2(accel.x, accel.z)
        self.current_angular_vel = msg.angular_velocity.y

    def joint_state_callback(self, msg):
        self.joint_positions = list(msg.position)

    def control_loop(self):
        if len(self.joint_positions) == 0:
            return

        # PD control for balance
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        target = self.get_parameter('target_angle').value

        error = target - self.current_angle
        d_error = -self.current_angular_vel

        control_effort = kp * error + kd * d_error

        # Apply to ankle joints (simplified)
        cmd = Float64MultiArray()
        cmd.data = [control_effort, control_effort]  # Both ankles

        self.joint_cmd_pub.publish(cmd)

def main():
    rclpy.init()
    controller = BalanceController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.10 Best Practices

### Code Organization
1. **One node, one purpose**: Keep nodes focused
2. **Use packages**: Group related functionality
3. **Namespaces**: Avoid topic name collisions
4. **Configuration files**: Use YAML for parameters

### Performance
1. **Choose appropriate QoS**: Match the use case
2. **Avoid unnecessary copies**: Use shared pointers in C++
3. **Control loop rates**: Match sensor rates
4. **Use composable nodes**: For tight coupling

### Debugging
1. **Logging levels**: Use DEBUG, INFO, WARN, ERROR appropriately
2. **Record bags**: Capture data for offline analysis
3. **Use rqt_console**: View logs from all nodes
4. **TF debugging**: Check with `ros2 run tf2_tools view_frames`

### Safety
1. **Timeouts**: Detect lost connections
2. **Watchdogs**: Monitor critical nodes
3. **Emergency stops**: Always have a way to stop
4. **Graceful shutdown**: Clean up resources

## Key Takeaways

1. ROS 2 is a powerful, industry-standard middleware for robotics
2. Core concepts: Nodes, topics, services, actions, parameters
3. Tools: CLI, RViz2, rqt for development and debugging
4. TF2 manages coordinate transformations between robot frames
5. QoS policies control communication reliability and performance
6. Modular design and proper organization are essential
7. Launch files coordinate complex multi-node systems

## Exercises

### Beginner
1. Install ROS 2 and create a simple "Hello World" node
2. Create a publisher and subscriber for a string message
3. Use ros2 topic echo to view live data
4. Launch multiple nodes with a launch file

### Intermediate
1. Create a custom message type for robot commands
2. Implement a service for emergency stop
3. Set up TF2 transforms for a simple robot
4. Record and playback sensor data using rosbags

### Advanced
1. Build a multi-node system with proper QoS configuration
2. Implement an action server for a long-running task
3. Create composable nodes for performance
4. Set up cross-machine communication over network

## Additional Resources

- **Official Docs**: [https://docs.ros.org/en/humble/](https://docs.ros.org/en/humble/)
- **Tutorials**: ROS 2 official tutorials (beginner to advanced)
- **Videos**: The Construct YouTube channel
- **Community**: ROS Discourse, ROS Answers
- **Books**: "A Gentle Introduction to ROS" by Jason M. O'Kane

---

**Next Chapter**: [Gazebo Simulation](/docs/chapter-03-gazebo)
