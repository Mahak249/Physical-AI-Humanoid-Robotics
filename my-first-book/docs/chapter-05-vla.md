---
sidebar_position: 6
---

# Chapter 5: Vision-Language-Action (VLA) Models

## Overview

Vision-Language-Action (VLA) models represent a paradigm shift in robot learning by combining visual perception, natural language understanding, and action generation in a single end-to-end learned model. This chapter explores how foundation models are revolutionizing humanoid robotics.

## 5.1 The Evolution of Robot Learning

### Traditional Approaches

:::tip Beginner Explanation
Imagine teaching a robot by showing it thousands of examples of picking up a cup. Traditional methods required separate systems for seeing (vision), understanding what to do (planning), and moving (control). VLA models combine all three into one intelligent system that can understand commands like "bring me that red cup."
:::

**Classical Pipeline (Pre-2020):**
```
Perception → State Estimation → Planning → Control → Actuation
    ↓             ↓                ↓          ↓
 Computer     Kalman           Motion     PID
  Vision      Filter           Planner   Controller
```

**Limitations:**
- Hand-engineered features
- Brittle to distribution shifts
- Requires expert knowledge for each component
- Difficult to transfer between tasks
- No language understanding

### The Foundation Model Era (2020-Present)

**VLA Pipeline:**
```
[Image + Language] → Foundation Model → Actions
         ↓                    ↓              ↓
    RGB Camera          Transformer     Joint Targets
    Depth Sensor       (Pre-trained)    End-effector
                                         Gripper
```

**Advantages:**
- End-to-end learning
- Natural language task specification
- Transfer learning across tasks and robots
- Leverages internet-scale pretraining
- Generalization to novel scenarios

## 5.2 What are VLA Models?

### Core Components

**1. Vision Encoder**
- Processes RGB(D) images
- Typically a Vision Transformer (ViT) or ResNet
- Pre-trained on large image datasets (ImageNet, CLIP)

**2. Language Encoder**
- Processes natural language instructions
- BERT, GPT, or LLaMA based
- Understands task descriptions and goals

**3. Action Decoder**
- Generates robot actions
- Transformer or diffusion-based
- Outputs joint positions, velocities, or end-effector poses

**4. Multi-modal Fusion**
- Combines vision and language
- Cross-attention mechanisms
- Shared latent representations

### VLA Model Architecture

```python
class VLAModel(nn.Module):
    def __init__(self, vision_encoder, language_encoder, action_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder     # E.g., ViT
        self.language_encoder = language_encoder # E.g., BERT
        self.action_decoder = action_decoder     # E.g., Transformer
        self.fusion = CrossAttentionFusion()

    def forward(self, image, language, robot_state):
        # Encode inputs
        vision_features = self.vision_encoder(image)
        language_features = self.language_encoder(language)

        # Fuse modalities
        fused_features = self.fusion(vision_features, language_features)

        # Add robot proprioception
        combined = torch.cat([fused_features, robot_state], dim=-1)

        # Decode actions
        actions = self.action_decoder(combined)

        return actions
```

## 5.3 Prominent VLA Models

### RT-1 (Robotics Transformer 1)

**Google DeepMind, 2022**

:::info Advanced Note
RT-1 uses an EfficientNet image encoder, FiLM (Feature-wise Linear Modulation) for language conditioning, and a TokenLearner for efficient sequence modeling. It was trained on 130k robot demonstrations across 700+ tasks.
:::

**Key Features:**
- 35M parameters
- Trained on real robot data
- 13 robots, 700+ tasks
- Generalizes to new objects and scenarios

**Architecture:**
```
Image → EfficientNet-B3 → TokenLearner
Language → Universal Sentence Encoder
        ↓
   FiLM layers
        ↓
   Transformer
        ↓
   Action tokens (7-dim: xyz + rpy + gripper)
```

### RT-2 (Robotics Transformer 2)

**Google DeepMind, 2023**

**Innovation**: Co-finetuning vision-language models (VLMs) on both web data and robot data.

**Key Features:**
- Built on PaLI-X and PaLM-E
- 55B parameters (PaLM-E variant)
- Better zero-shot generalization
- Reasoning about novel situations

**Performance:**
- 3x better generalization than RT-1
- Emergent capabilities (e.g., "pick up extinct animal" → selects toy dinosaur)
- Transfers knowledge from web to robot

### Octo

**UC Berkeley, 2024**

**Innovation**: Open-source generalist robot policy trained on 800k robot trajectories from the Open X-Embodiment dataset.

**Key Features:**
- Diffusion-based action prediction
- Supports multiple robot morphologies
- Fine-tunable with as few as 50 demonstrations
- Flexible action spaces

**Architecture:**
```python
# Simplified Octo architecture
class OctoPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = ViT(patch_size=16)
        self.language = T5Encoder()
        self.diffusion_decoder = DiffusionPolicy(
            horizon=10,  # Predict 10 future actions
            action_dim=7
        )

    def forward(self, obs, language, timestep):
        # Encode observations
        vision_tokens = self.vision(obs['image'])
        lang_tokens = self.language(language)

        # Conditional diffusion
        # Starting from noise, iteratively denoise to get actions
        actions = self.diffusion_decoder.sample(
            condition=torch.cat([vision_tokens, lang_tokens]),
            num_steps=20
        )

        return actions
```

### OpenVLA

**Open-source VLA, 2024**

**Innovation**: Fully open-source VLA model with permissive licensing.

**Key Features:**
- 7B parameters
- LLaMA 2 backbone
- Trained on Open X-Embodiment
- Easy to fine-tune and deploy

## 5.4 Training VLA Models

### Data Collection

**1. Teleoperation**
```python
import rclpy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose
import h5py

class DataCollector:
    def __init__(self):
        self.episodes = []
        self.current_episode = {
            'observations': [],
            'actions': [],
            'language': ""
        }

    def record_step(self, image, robot_state, action):
        self.current_episode['observations'].append({
            'image': image,
            'robot_state': robot_state
        })
        self.current_episode['actions'].append(action)

    def save_episode(self, language_instruction):
        self.current_episode['language'] = language_instruction
        self.episodes.append(self.current_episode)
        self.current_episode = {
            'observations': [],
            'actions': [],
            'language': ""
        }

    def save_dataset(self, filename):
        with h5py.File(filename, 'w') as f:
            for i, episode in enumerate(self.episodes):
                grp = f.create_group(f'episode_{i}')
                grp.create_dataset('images', data=episode['observations'])
                grp.create_dataset('actions', data=episode['actions'])
                grp.attrs['language'] = episode['language']
```

**2. Simulation**
```python
# Generate synthetic data in Isaac Sim or Gazebo
for task in tasks:
    env.reset()
    language_instruction = task['instruction']  # "pick up the red block"

    for step in range(max_steps):
        observation = env.get_observation()  # Image + robot state
        action = expert_policy(observation, language_instruction)

        env.step(action)
        collector.record_step(observation, action)

    collector.save_episode(language_instruction)
```

### Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class VLATrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()  # For action prediction

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            images = batch['image'].to(device)
            language = batch['language']
            robot_state = batch['robot_state'].to(device)
            actions_gt = batch['actions'].to(device)

            # Forward pass
            actions_pred = self.model(images, language, robot_state)

            # Compute loss
            loss = self.criterion(actions_pred, actions_gt)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(device)
                language = batch['language']
                robot_state = batch['robot_state'].to(device)
                actions_gt = batch['actions'].to(device)

                actions_pred = self.model(images, language, robot_state)
                loss = self.criterion(actions_pred, actions_gt)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")

            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(),
                          f"vla_checkpoint_{epoch}.pt")
```

### Fine-tuning for Your Robot

```python
from transformers import AutoModel
import torch

# Load pre-trained VLA model (e.g., Octo or OpenVLA)
pretrained_model = AutoModel.from_pretrained("openvla-7b")

# Freeze vision and language encoders
for param in pretrained_model.vision_encoder.parameters():
    param.requires_grad = False
for param in pretrained_model.language_encoder.parameters():
    param.requires_grad = False

# Fine-tune only action decoder
for param in pretrained_model.action_decoder.parameters():
    param.requires_grad = True

# Train on your robot's data (even just 50-100 demonstrations)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, pretrained_model.parameters()),
    lr=1e-5
)

# Fine-tuning loop
for epoch in range(num_epochs):
    for batch in your_robot_dataloader:
        loss = train_step(pretrained_model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 5.5 Deploying VLA Models

### Inference on Robot

```python
import torch
import cv2
import numpy as np
from transformers import AutoModel, AutoTokenizer

class VLARobotController:
    def __init__(self, model_name, device='cuda'):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def preprocess_image(self, image):
        # Resize and normalize
        image = cv2.resize(image, (224, 224))
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image / 255.0
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return image.unsqueeze(0).to(self.device)

    def preprocess_language(self, instruction):
        tokens = self.tokenizer(
            instruction,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def get_action(self, image, instruction, robot_state):
        with torch.no_grad():
            # Preprocess inputs
            image_tensor = self.preprocess_image(image)
            language_tokens = self.preprocess_language(instruction)
            state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(self.device)

            # Inference
            action = self.model(
                image=image_tensor,
                language=language_tokens,
                robot_state=state_tensor
            )

            return action.cpu().numpy()[0]

# Usage with ROS 2
class VLANode(Node):
    def __init__(self):
        super().__init__('vla_controller')

        self.vla = VLARobotController('openvla-7b')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )

        # State
        self.current_image = None
        self.robot_state = None
        self.instruction = "pick up the red block"  # From speech/GUI

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def image_callback(self, msg):
        self.current_image = bridge.imgmsg_to_cv2(msg, 'rgb8')

    def joint_state_callback(self, msg):
        self.robot_state = np.array(msg.position)

    def control_loop(self):
        if self.current_image is None or self.robot_state is None:
            return

        # Get action from VLA model
        action = self.vla.get_action(
            self.current_image,
            self.instruction,
            self.robot_state
        )

        # Publish action
        cmd = Float64MultiArray()
        cmd.data = action.tolist()
        self.cmd_pub.publish(cmd)
```

### Optimization for Real-Time Performance

**1. Model Quantization**
```python
import torch

# Post-training quantization
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Reduces model size by ~4x, speeds up inference by 2-3x
```

**2. TensorRT Optimization**
```python
import torch_tensorrt

# Compile model for NVIDIA GPU
trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input((1, 3, 224, 224)),  # Image
        torch_tensorrt.Input((1, 512)),          # Language
        torch_tensorrt.Input((1, 21))            # Robot state
    ],
    enabled_precisions={torch.float16},  # FP16 inference
)

# 2-5x speedup on RTX GPUs
```

**3. ONNX Export**
```python
import torch.onnx

# Export to ONNX for cross-platform deployment
torch.onnx.export(
    model,
    (dummy_image, dummy_language, dummy_state),
    "vla_model.onnx",
    input_names=['image', 'language', 'robot_state'],
    output_names=['actions'],
    dynamic_axes={
        'image': {0: 'batch'},
        'language': {0: 'batch'},
        'robot_state': {0: 'batch'},
        'actions': {0: 'batch'}
    }
)
```

## 5.6 Advanced Topics

### Diffusion Policies

Diffusion models for action generation offer better multi-modal behavior modeling.

```python
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim, horizon):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.noise_predictor = UNet1D(
            input_dim=action_dim,
            cond_dim=512  # Conditioning from vision+language
        )

    def forward(self, x_t, t, condition):
        # Predict noise at timestep t
        return self.noise_predictor(x_t, t, condition)

    def sample(self, condition, num_steps=20):
        # Start from random noise
        x = torch.randn((1, self.horizon, self.action_dim))

        # Iterative denoising
        for t in reversed(range(num_steps)):
            noise_pred = self(x, t, condition)
            x = denoise_step(x, noise_pred, t)

        return x
```

### Hierarchical VLA

Combining high-level language understanding with low-level control:

```python
class HierarchicalVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_level_planner = LanguageToGoal()  # LLM-based
        self.low_level_controller = VLAModel()      # Action policy

    def forward(self, observation, language_instruction):
        # High-level: Language → Subgoals
        subgoals = self.high_level_planner(language_instruction)

        # Low-level: Vision + Subgoal → Actions
        actions = []
        for subgoal in subgoals:
            action = self.low_level_controller(observation, subgoal)
            actions.append(action)

        return actions
```

### Multi-Robot Coordination

```python
class MultiRobotVLA(nn.Module):
    def __init__(self, num_robots):
        super().__init__()
        self.individual_policies = nn.ModuleList([
            VLAModel() for _ in range(num_robots)
        ])
        self.coordinator = TransformerEncoder()  # Coordinate between robots

    def forward(self, observations, language):
        # Each robot's individual policy
        individual_actions = [
            policy(obs, language)
            for policy, obs in zip(self.individual_policies, observations)
        ]

        # Coordinate
        coordinated_actions = self.coordinator(individual_actions)

        return coordinated_actions
```

## 5.7 Challenges and Future Directions

### Current Limitations

1. **Data Hungry**: Requires large datasets (100k+ demonstrations)
2. **Sim-to-Real Gap**: Simulation training doesn't always transfer
3. **Safety**: No formal guarantees on behavior
4. **Interpretability**: Black-box nature
5. **Latency**: Large models can be slow
6. **Generalization**: Still struggles with truly novel scenarios

### Emerging Solutions

**1. Few-Shot Learning**
- Meta-learning approaches
- Prompt-based adaptation
- In-context learning (GPT-style)

**2. Self-Supervised Learning**
- Learn from robot's own experience
- Curiosity-driven exploration
- World models

**3. Formal Verification**
- Safety shields
- Runtime monitoring
- Certified robustness

**4. Explainable AI**
- Attention visualization
- Concept-based explanations
- Natural language reasoning

## Key Takeaways

1. VLA models unify vision, language, and action in end-to-end learning
2. Foundation models enable transfer learning across tasks and robots
3. Pre-trained VLAs (RT-2, Octo, OpenVLA) can be fine-tuned with limited data
4. Natural language enables intuitive human-robot interaction
5. Deployment requires optimization (quantization, TensorRT, ONNX)
6. Active research area with rapid progress
7. Combination of classical control and learned policies often works best

## Exercises

### Beginner
1. Understand the difference between traditional and VLA approaches
2. Explore pre-trained VLA models (Octo, OpenVLA)
3. Visualize attention maps from a VLA model
4. Test language variations on robot tasks

### Intermediate
1. Collect teleoperation data for a simple task
2. Fine-tune a pre-trained VLA on your data
3. Deploy VLA model in simulation (Gazebo/Isaac Sim)
4. Optimize model for real-time inference

### Advanced
1. Implement a diffusion-based action policy
2. Train a VLA model from scratch
3. Deploy on real robot hardware
4. Combine VLA with classical motion planning

## Additional Resources

- **Papers**:
  - "RT-1: Robotics Transformer for Real-World Control at Scale"
  - "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
  - "Octo: An Open-Source Generalist Robot Policy"
- **Code**:
  - OpenVLA: https://github.com/openvla/openvla
  - Octo: https://github.com/octo-models/octo
- **Datasets**:
  - Open X-Embodiment: https://robotics-transformer-x.github.io/
- **Community**: Discord servers for robotics ML, ROS 2

---

**Next Chapter**: [Humanoid Robotics Hardware](/docs/chapter-06-hardware)
