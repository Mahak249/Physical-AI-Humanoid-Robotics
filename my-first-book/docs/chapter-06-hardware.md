---
sidebar_position: 7
---

# Chapter 6: Humanoid Robotics Hardware

## Overview

Building or working with humanoid robots requires understanding the physical components that bring intelligence to life. This chapter covers actuators, sensors, embedded systems, power management, and integration strategies for humanoid platforms.

## 6.1 Actuators: The Muscles of Humanoid Robots

### Types of Actuators

:::tip Beginner Explanation
Actuators are like the muscles in your body - they make parts of the robot move. Just like we have different types of muscles for different tasks (strong for lifting, precise for writing), robots use different types of actuators depending on what they need to do.
:::

**1. Electric Motors**

**DC Brushed Motors**
- Simple, cheap, widely available
- Good for prototypes and learning
- Lower efficiency, requires maintenance (brushes wear out)
- Used in hobby servos

**DC Brushless Motors (BLDC)**
- High efficiency (85-90%)
- No brush maintenance
- Better power-to-weight ratio
- Requires electronic speed controller (ESC)
- **Common in humanoids**: High-performance joints

**Servo Motors**
- Built-in position control
- Feedback mechanism (potentiometer or encoder)
- Easy to control (PWM signal)
- Limited torque and range
- **Common in humanoids**: Research platforms, smaller robots

**2. Hydraulic Actuators**

**Advantages:**
- Extremely high force/torque
- Good power-to-weight ratio
- Smooth motion

**Disadvantages:**
- Complex system (pumps, valves, fluid)
- Maintenance intensive
- Risk of leaks
- Noisy operation

**Examples**: Boston Dynamics Atlas (early versions)

**3. Pneumatic Actuators**

**Advantages:**
- Compliant (soft, safe)
- Fast response
- Simple design

**Disadvantages:**
- Low precision
- Requires air compressor
- Noisy
- Difficult to control accurately

**Examples**: Soft robotics, prosthetics

**4. Series Elastic Actuators (SEA)**

:::info Advanced Note
SEAs place a compliant element (spring) between the motor and load, providing:
- Force sensing via spring deflection
- Shock absorption
- Safe human interaction
- Better force control
:::

**Design:**
```
Motor → Gearbox → Spring → Output Link
                    ↓
              Force Sensor
```

**Examples**: NASA Valkyrie, many modern humanoids

**5. Quasi-Direct Drive (QDD) Actuators**

**Innovation**: Low gear ratio (6:1 to 9:1) instead of typical 50:1+

**Advantages:**
- High backdrivability
- Accurate force sensing
- Better dynamic performance
- Shock tolerance

**Disadvantages:**
- Requires high-torque motors
- More expensive
- Thermal management challenges

**Examples**: MIT Cheetah, Boston Dynamics Atlas (recent)

### Actuator Selection Criteria

| **Criterion**           | **Joints (Hip, Knee)** | **Joints (Ankle, Wrist)** | **Fingers/Hands**    |
|------------------------|------------------------|---------------------------|----------------------|
| **Torque**             | High (50-200 Nm)       | Medium (10-50 Nm)         | Low (0.5-5 Nm)       |
| **Speed**              | Medium                 | Medium-High               | High                 |
| **Precision**          | Medium                 | High                      | Very High            |
| **Backdrivability**    | Desirable              | Very Important            | Critical             |
| **Cost**               | Can be high            | Medium                    | Should be low        |

### Actuator Control

**Voltage Control (Simple)**
```python
# Set motor voltage (open-loop)
motor.set_voltage(12.0)  # Volts
```

**Position Control (PID)**
```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.prev_error = 0

    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

# Usage
pid = PIDController(kp=50.0, ki=0.1, kd=5.0)
target_angle = 1.57  # radians (90 degrees)
current_angle = encoder.read()
voltage = pid.compute(target_angle, current_angle, dt=0.01)
motor.set_voltage(voltage)
```

**Torque Control**
```python
# Direct torque control (requires current sensing)
target_torque = 10.0  # Nm
current_command = target_torque / motor.kt  # Motor torque constant
motor.set_current(current_command)
```

**Impedance Control**
```python
# Compliant control: F = K * (x - x_target) + B * (v - v_target)
def impedance_control(position, velocity, target_pos, target_vel, K, B):
    """
    K: Stiffness (N/m or Nm/rad)
    B: Damping (Ns/m or Nms/rad)
    """
    force = K * (target_pos - position) + B * (target_vel - velocity)
    return force

# For joint control
K_joint = 100  # Nm/rad
B_joint = 10   # Nms/rad
torque = impedance_control(joint_angle, joint_velocity,
                          target_angle, target_velocity, K_joint, B_joint)
motor.set_torque(torque)
```

## 6.2 Sensors: The Senses of Humanoid Robots

### Proprioceptive Sensors (Internal State)

**1. Encoders**

**Incremental Encoders:**
- Count pulses for relative position
- High resolution (1000-10000 CPR)
- Requires homing/calibration

**Absolute Encoders:**
- Know position on power-up
- No homing required
- More expensive

```python
# Reading encoder
encoder_counts = encoder.read()
angle_rad = (encoder_counts / encoder.resolution) * 2 * math.pi
```

**2. Current Sensors**

Measure motor current to estimate torque:
```python
current = current_sensor.read()  # Amps
torque_estimate = current * motor.kt  # Motor torque constant
```

**3. Inertial Measurement Unit (IMU)**

**Components:**
- 3-axis accelerometer
- 3-axis gyroscope
- (Optional) 3-axis magnetometer

```python
import smbus
import time

class IMU:
    def __init__(self, i2c_bus=1, address=0x68):
        self.bus = smbus.SMBus(i2c_bus)
        self.address = address
        # Initialize IMU (e.g., MPU6050)
        self.bus.write_byte_data(self.address, 0x6B, 0)

    def read_accel(self):
        # Read 6 bytes starting from register 0x3B
        data = self.bus.read_i2c_block_data(self.address, 0x3B, 6)
        x = self.convert(data[0], data[1])
        y = self.convert(data[2], data[3])
        z = self.convert(data[4], data[5])
        return x, y, z

    def read_gyro(self):
        data = self.bus.read_i2c_block_data(self.address, 0x43, 6)
        x = self.convert(data[0], data[1])
        y = self.convert(data[2], data[3])
        z = self.convert(data[4], data[5])
        return x, y, z

    def convert(self, high, low):
        value = (high << 8) | low
        if value > 32768:
            value -= 65536
        return value

# Usage
imu = IMU()
ax, ay, az = imu.read_accel()
gx, gy, gz = imu.read_gyro()
print(f"Accel: ({ax}, {ay}, {az})")
print(f"Gyro: ({gx}, {gy}, {gz})")
```

**4. Force-Torque Sensors**

Measure contact forces at feet, hands, or joints:
```python
ft_sensor = ForceTorqueSensor('/dev/ttyUSB0')
forces, torques = ft_sensor.read()
print(f"Force (x,y,z): {forces}")
print(f"Torque (x,y,z): {torques}")

# Use for contact detection
if forces[2] > 50.0:  # Z-force (normal to ground)
    print("Foot in contact with ground")
```

### Exteroceptive Sensors (External Environment)

**1. Cameras**

**Types:**
- RGB camera: Color images
- Depth camera: Distance to objects
- Stereo camera: Two cameras for depth via disparity
- Event camera: Asynchronous pixel-level changes

```python
import cv2

# RGB camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('Camera', frame)

# Depth camera (e.g., Intel RealSense)
import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())
```

**2. LIDAR**

3D scanning for mapping and obstacle avoidance:
```python
import rplidar
lidar = rplidar.RPLidar('/dev/ttyUSB0')

for scan in lidar.iter_scans():
    for (quality, angle, distance) in scan:
        print(f"Angle: {angle:.2f}°, Distance: {distance:.2f}mm")
```

**3. Tactile Sensors**

For grasping and manipulation:
- Resistive: Pressure → Resistance change
- Capacitive: Touch → Capacitance change
- Piezoelectric: Force → Voltage

```python
# Example: Reading tactile sensor array
tactile_grid = tactile_sensor.read_grid()  # 16x16 pressure values
max_pressure = np.max(tactile_grid)
contact_location = np.unravel_index(np.argmax(tactile_grid), tactile_grid.shape)
print(f"Max pressure: {max_pressure} at {contact_location}")
```

## 6.3 Embedded Computing

### Computation Platforms

**1. Microcontrollers (MCU)**

**Use Cases:** Low-level motor control, sensor reading

**Popular Choices:**
- STM32 (ARM Cortex-M series)
- Teensy 4.1 (ARM Cortex-M7, 600 MHz)
- Arduino (for prototyping)

**Example: Joint Controller on Teensy**
```cpp
// Teensy C++ code
#include <Encoder.h>
#include <Arduino.h>

Encoder encoder(2, 3);
int motor_pwm_pin = 9;

float kp = 50.0;
float kd = 5.0;
float target_position = 0.0;
float previous_error = 0.0;

void setup() {
  pinMode(motor_pwm_pin, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  // Read encoder
  long encoder_counts = encoder.read();
  float position = (encoder_counts / 2000.0) * 2.0 * PI;

  // PD control
  float error = target_position - position;
  float velocity = (error - previous_error) / 0.01;  // 100 Hz loop
  float control = kp * error + kd * velocity;

  // Output PWM
  int pwm_value = constrain(control, -255, 255);
  analogWrite(motor_pwm_pin, abs(pwm_value));
  digitalWrite(motor_dir_pin, pwm_value > 0 ? HIGH : LOW);

  previous_error = error;
  delay(10);  // 100 Hz
}
```

**2. Single-Board Computers (SBC)**

**Use Cases:** High-level planning, vision, AI inference

**Popular Choices:**
- Raspberry Pi 4/5 (ARM Cortex-A72, 1.5-2.4 GHz, 4-8 GB RAM)
- NVIDIA Jetson (Orin Nano, Xavier, AGX)
- Intel NUC (x86, more powerful)

**Raspberry Pi Setup:**
```bash
# Install ROS 2
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install ros-humble-ros-base

# Run ROS 2 node on Pi
source /opt/ros/humble/setup.bash
ros2 run my_package humanoid_controller
```

**NVIDIA Jetson for AI:**
```bash
# Install PyTorch
pip3 install torch torchvision

# Run VLA model
python3 vla_controller.py  # Runs on Jetson GPU
```

**3. GPU Compute Modules**

**For on-board AI:** NVIDIA Jetson series

**Performance:**
- Jetson Orin Nano: 40 TOPS (int8), ~$500
- Jetson Orin NX: 100 TOPS, ~$800
- Jetson AGX Orin: 275 TOPS, ~$2000

```python
# Deploy TensorRT model on Jetson
import tensorrt as trt
import pycuda.driver as cuda

# Load TensorRT engine
with open("vla_model.trt", "rb") as f:
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Inference
input_data = preprocess(image)
cuda.memcpy_htod(d_input, input_data)
context.execute_v2([d_input, d_output])
cuda.memcpy_dtoh(output, d_output)
actions = postprocess(output)
```

### System Architecture

**Hierarchical Control:**
```
┌─────────────────────────────────────┐
│     High-Level PC (Off-board)       │ ← VLA models, Planning
│   Intel i9 + NVIDIA RTX 4090        │
└──────────────┬──────────────────────┘
               │ Ethernet / WiFi
┌──────────────▼──────────────────────┐
│     Mid-Level SBC (On-board)        │ ← Sensor processing
│   NVIDIA Jetson Orin / Raspberry Pi │   State estimation
└──────────────┬──────────────────────┘   ROS 2 nodes
               │ CAN / Ethernet
┌──────────────▼──────────────────────┐
│   Low-Level MCU (On each joint)     │ ← Motor control (1kHz)
│   STM32 / Teensy                    │   Safety checks
└─────────────────────────────────────┘   Current/position control
```

## 6.4 Power Systems

### Power Requirements

**Typical Humanoid Power Budget:**
```
Component              | Power (W)  | Voltage
-----------------------|------------|----------
Compute (Jetson Orin)  | 15-60      | 12-19V
Motors (20x)           | 500-2000   | 24-48V
Sensors (cameras, IMU) | 10-30      | 5-12V
Total Peak             | 1000-3000W |
```

### Battery Technologies

**Lithium Polymer (LiPo)**
- High energy density (150-250 Wh/kg)
- High discharge rate (30C-50C)
- Requires careful charging/monitoring
- Common in research humanoids

**Lithium-ion (Li-ion)**
- Good energy density (100-200 Wh/kg)
- Safer than LiPo
- Longer cycle life
- Used in commercial robots

**Battery Configuration:**
```
Voltage = 48V (12S LiPo)
Capacity = 20Ah
Energy = 48V × 20Ah = 960 Wh

Runtime at 500W avg power:
960 Wh / 500 W = 1.92 hours
```

### Power Distribution

```python
# Battery Management System (BMS) monitoring
class BatteryMonitor:
    def __init__(self, serial_port):
        self.bms = BMS(serial_port)
        self.voltage_threshold = 42.0  # Volts (for 12S)
        self.current_limit = 60.0  # Amps

    def monitor(self):
        voltage = self.bms.read_voltage()
        current = self.bms.read_current()
        soc = self.bms.read_state_of_charge()  # State of Charge (%)

        print(f"Voltage: {voltage}V, Current: {current}A, SOC: {soc}%")

        # Safety checks
        if voltage < self.voltage_threshold:
            self.emergency_shutdown("Low battery voltage")

        if current > self.current_limit:
            self.reduce_power()

    def emergency_shutdown(self, reason):
        print(f"Emergency shutdown: {reason}")
        # Send stop commands to all motors
        # Activate brakes
        # Log data

    def reduce_power(self):
        # Reduce motor torque limits
        # Decrease sensor sampling rates
        pass
```

### Voltage Regulation

```
48V Battery
   ├─→ DC-DC (48V → 24V, 30A) → Motor controllers
   ├─→ DC-DC (48V → 12V, 10A) → Sensors, fans
   └─→ DC-DC (48V → 5V, 5A)   → Logic level, MCUs
```

## 6.5 Communication Protocols

### CAN Bus (Controller Area Network)

**Advantages:**
- Reliable in noisy environments
- Multi-master
- Priority-based arbitration
- Widely used in robotics

```python
import can

# Setup CAN interface
bus = can.interface.Bus(channel='can0', bustype='socketcan')

# Send motor command
msg = can.Message(
    arbitration_id=0x123,  # Motor controller ID
    data=[0x01, 0x10, 0x00, 0x50],  # Command + target position
    is_extended_id=False
)
bus.send(msg)

# Receive sensor data
message = bus.recv(timeout=0.1)
if message:
    print(f"Received: {message.data}")
```

### EtherCAT (Ethernet for Control Automation Technology)

**Advantages:**
- Very high speed (100 Mbps - 1 Gbps)
- Deterministic (&lt;100 μs cycle time)
- Large data payloads
- Used in industrial humanoids

```python
import pysoem

# EtherCAT master setup
master = pysoem.Master()
master.open('eth0')

# Configure slaves (motor controllers)
master.config_init()

# Cyclic operation
while True:
    master.send_processdata()
    master.receive_processdata(timeout=2000)

    # Read/write to slaves
    motor1_position = master.slaves[0].input  # Read
    master.slaves[0].output = target_position  # Write

    time.sleep(0.001)  # 1 kHz
```

### RS-485 / TTL Serial

**Advantages:**
- Simple, low-cost
- Good for daisy-chaining sensors
- Half-duplex communication

```python
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)

# Dynamixel servo example (TTL)
def set_goal_position(servo_id, position):
    # Dynamixel Protocol 2.0 packet
    packet = [0xFF, 0xFF, 0xFD, 0x00,  # Header
              servo_id,  # Servo ID
              0x07, 0x00,  # Length
              0x03,  # Instruction: Write
              0x74, 0x00,  # Address: Goal Position
              position & 0xFF, (position >> 8) & 0xFF,
              (position >> 16) & 0xFF, (position >> 24) & 0xFF]

    # Add checksum
    checksum = (~sum(packet[5:]) & 0xFFFF)
    packet.extend([checksum & 0xFF, (checksum >> 8) & 0xFF])

    ser.write(bytes(packet))

set_goal_position(servo_id=1, position=2048)  # Center position
```

## 6.6 Mechanical Design Considerations

### Materials

**Aluminum:**
- Lightweight, machinable
- Good strength-to-weight
- Used in frames and links

**Carbon Fiber:**
- Very high strength-to-weight
- Expensive, requires specialized manufacturing
- Used in advanced platforms

**3D Printed Plastics (PLA, ABS, Nylon):**
- Rapid prototyping
- Complex geometries
- Lower strength
- Used in prototypes, non-load-bearing parts

**Titanium:**
- Extremely strong and lightweight
- Very expensive
- Used in critical joints (hip, knee)

### Joint Design

**Revolute Joint (Hinge):**
```
Motor + Gearbox + Bearing + Encoder
```

**Example: Knee Joint**
- Motor: BLDC 200W
- Gearbox: 6:1 planetary
- Torque: ~60 Nm
- Range: 0° to 150°
- Encoder: 14-bit absolute

### Safety Mechanisms

**1. Mechanical Hard Stops**
- Prevent over-extension
- Protect joints from damage

**2. Torque Limiting**
```python
MAX_TORQUE = 50.0  # Nm

def safe_torque_control(target_torque):
    limited_torque = np.clip(target_torque, -MAX_TORQUE, MAX_TORQUE)
    motor.set_torque(limited_torque)
```

**3. Emergency Stop**
```python
class EmergencyStop:
    def __init__(self, gpio_pin):
        self.estop_button = GPIO(gpio_pin, GPIO.IN)

    def check(self):
        if self.estop_button.read() == 0:  # Button pressed
            self.trigger_estop()

    def trigger_estop(self):
        # Cut power to all motors
        # Activate mechanical brakes
        # Log state
        # Notify operator
        print("EMERGENCY STOP ACTIVATED")
        motor_controller.disable_all()
```

**4. Watchdog Timer**
```python
import threading

class Watchdog:
    def __init__(self, timeout=0.1):
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def monitor(self):
        while self.running:
            if time.time() - self.last_heartbeat > self.timeout:
                print("Watchdog timeout - stopping robot")
                motor_controller.emergency_stop()
            time.sleep(0.01)

# Usage in control loop
watchdog = Watchdog(timeout=0.1)
while True:
    control_step()
    watchdog.heartbeat()  # Must be called regularly
```

## 6.7 Practical Example: Building a Simple Biped

### Bill of Materials (BOM)

| Component                    | Quantity | Est. Cost |
|------------------------------|----------|-----------|
| Dynamixel MX-106 servos      | 12       | $3,600    |
| Raspberry Pi 4 (8GB)         | 1        | $80       |
| IMU (BNO055)                 | 1        | $30       |
| USB Camera                   | 1        | $40       |
| LiPo Battery (6S, 5000mAh)   | 2        | $200      |
| DC-DC converters             | 3        | $60       |
| Aluminum frame/brackets      | -        | $300      |
| 3D printed parts             | -        | $100      |
| Cables, connectors, misc     | -        | $150      |
| **Total**                    |          | **~$4,560** |

### Assembly Steps

1. **Frame Construction**
   - CNC or 3D print body parts
   - Assemble with bolts and brackets

2. **Actuator Installation**
   - Mount servos in joints
   - Ensure proper alignment
   - Secure with screws and locktite

3. **Wiring**
   - Daisy-chain servos (TTL/RS-485)
   - Connect to Raspberry Pi
   - Install power distribution board
   - Add fuses and switches

4. **Sensor Integration**
   - Mount IMU on torso
   - Install cameras in head
   - Connect foot force sensors (if using)

5. **Software Setup**
   - Install ROS 2 on Raspberry Pi
   - Configure servo driver node
   - Calibrate IMU and joints
   - Test individual joints

### Basic Walking Controller

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class SimpleBipedWalker(Node):
    def __init__(self):
        super().__init__('simple_biped_walker')

        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Joint names (12 DOF biped)
        self.joints = [
            'left_hip_roll', 'left_hip_pitch', 'left_knee',
            'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_roll', 'right_hip_pitch', 'right_knee',
            'right_ankle_pitch', 'right_ankle_roll',
            'torso_pitch', 'torso_yaw'
        ]

        self.joint_positions = np.zeros(12)
        self.time = 0.0

        # Control loop at 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

    def joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)

    def control_loop(self):
        self.time += 0.02
        phase = (self.time % 2.0) / 2.0  # 2-second gait cycle

        # Simple sinusoidal gait pattern
        target_positions = np.zeros(12)

        if phase < 0.5:
            # Left leg stance, right leg swing
            target_positions[1] = -0.3 * (phase / 0.5)  # Left hip back
            target_positions[2] = 0.2  # Left knee slight bend
            target_positions[7] = 0.6 * np.sin(phase * 2 * np.pi)  # Right knee swing
        else:
            # Right leg stance, left leg swing
            swing_phase = (phase - 0.5) / 0.5
            target_positions[6] = -0.3 * swing_phase  # Right hip back
            target_positions[7] = 0.2  # Right knee slight bend
            target_positions[2] = 0.6 * np.sin(swing_phase * 2 * np.pi)  # Left knee swing

        # Torso compensation
        target_positions[10] = 0.1 * np.sin(phase * 4 * np.pi)  # Torso pitch

        # Publish commands
        cmd = Float64MultiArray()
        cmd.data = target_positions.tolist()
        self.joint_pub.publish(cmd)

def main():
    rclpy.init()
    walker = SimpleBipedWalker()
    rclpy.spin(walker)
    walker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

1. Actuators (motors, hydraulics) are critical for humanoid motion
2. Sensors (IMU, encoders, cameras, force) provide state awareness
3. Embedded systems (MCU + SBC + GPU) form the computational hierarchy
4. Power management ensures safe operation and adequate runtime
5. Communication protocols (CAN, EtherCAT) enable real-time control
6. Mechanical design affects performance, safety, and cost
7. Safety systems (e-stop, watchdog, limits) are non-negotiable
8. Integration requires systems thinking across hardware and software

## Exercises

### Beginner
1. Calculate torque requirements for a humanoid knee joint
2. Select appropriate motors for a 50 kg humanoid
3. Design a simple power distribution system
4. Set up a Raspberry Pi with ROS 2 for robot control

### Intermediate
1. Implement PID control for a real servo motor
2. Interface an IMU with a microcontroller
3. Design a 3D-printed humanoid leg structure
4. Set up CAN bus communication between devices

### Advanced
1. Build a complete biped robot from scratch
2. Implement whole-body balance control
3. Integrate VLA model with real hardware
4. Design a custom brushless motor controller

## Additional Resources

- **Hardware**:
  - Dynamixel servos documentation
  - NVIDIA Jetson developer guides
  - STM32 reference manuals
- **Mechanical**:
  - "Robot Builder's Bonanza" by Gordon McComb
  - Fusion 360 / SolidWorks tutorials
- **Electrical**:
  - "Practical Electronics for Inventors" by Paul Scherz
  - Motor control tutorials (SimpleFOC, ODrive)
- **Safety Standards**:
  - ISO 13482 (Safety requirements for personal care robots)
  - IEC 61508 (Functional safety)

---

**Congratulations!** You've completed the main chapters of the Physical AI & Humanoid Robotics textbook. Continue with advanced topics, projects, and real-world applications in the appendices.
