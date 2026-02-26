# GEAR-SONIC ONNX Models — Python Deployment Guide

This document describes the three ONNX models in the GEAR-SONIC deployment pipeline,
their exact tensor specifications, physical meanings, and how to run them from Python
using `onnxruntime`.

## Pipeline Overview

```
User Input (VR / gamepad / command)
        |
        v
  Planner (10 Hz)              <-- planner_sonic.onnx
  Generates reference trajectory (30 Hz poses)
        |
        v
  Encoder (50 Hz)              <-- model_encoder.onnx
  Compresses motion reference + sensor obs --> 64-dim latent tokens
        |
        v
  Policy / Decoder (50 Hz)     <-- model_decoder.onnx
  (tokens + robot state history) --> 29 motor commands
        |
        v
  Motor Driver (500 Hz PD control)
```

---

## Joint Ordering

Two joint orderings coexist. All planner I/O uses **MuJoCo order**. The policy
output `action` is in **IsaacLab order**.

### MuJoCo Order (29 joints, used by planner `qpos[7:36]`)

| Index | Joint Name |
|-------|------------|
| 0 | left_hip_pitch |
| 1 | left_hip_roll |
| 2 | left_hip_yaw |
| 3 | left_knee |
| 4 | left_ankle_pitch |
| 5 | left_ankle_roll |
| 6 | right_hip_pitch |
| 7 | right_hip_roll |
| 8 | right_hip_yaw |
| 9 | right_knee |
| 10 | right_ankle_pitch |
| 11 | right_ankle_roll |
| 12 | waist_yaw |
| 13 | waist_roll |
| 14 | waist_pitch |
| 15 | left_shoulder_pitch |
| 16 | left_shoulder_roll |
| 17 | left_shoulder_yaw |
| 18 | left_elbow |
| 19 | left_wrist_roll |
| 20 | left_wrist_pitch |
| 21 | left_wrist_yaw |
| 22 | right_shoulder_pitch |
| 23 | right_shoulder_roll |
| 24 | right_shoulder_yaw |
| 25 | right_elbow |
| 26 | right_wrist_roll |
| 27 | right_wrist_pitch |
| 28 | right_wrist_yaw |

### Mapping Arrays

```python
import numpy as np

# isaaclab_to_mujoco[isaaclab_idx] = mujoco_idx
isaaclab_to_mujoco = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
    2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28
])

# mujoco_to_isaaclab[mujoco_idx] = isaaclab_idx
mujoco_to_isaaclab = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18,
    25, 19, 26, 20, 27, 21, 28
])
```

---

## 1. Planner — `planner_sonic.onnx`

**Path:** `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`

Generates a reference locomotion trajectory given a high-level command (mode,
direction, speed, height). Fully self-contained — the easiest model to use
standalone.

### Inputs (11 tensors)

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `context_mujoco_qpos` | `[1, 4, 36]` | float32 | 4 recent pose frames. Per frame: `[0:3]` root xyz (meters), `[3:7]` root quaternion (w,x,y,z), `[7:36]` 29 joint angles (radians) in MuJoCo order |
| `mode` | `[1]` | int64 | Locomotion mode (see table below) |
| `target_vel` | `[1]` | float32 | Target speed in m/s. Use -1.0 for mode default |
| `movement_direction` | `[1, 3]` | float32 | Unit vector (x,y,z) for movement direction. MuJoCo Z-up: X=forward, Y=left, Z=up |
| `facing_direction` | `[1, 3]` | float32 | Unit vector (x,y,z) for heading. Independent of movement (allows strafing) |
| `height` | `[1]` | float32 | Target pelvis height in meters. Use -1.0 for mode default (~0.789 m standing) |
| `random_seed` | `[1]` | int64 | Seed for motion stochasticity |
| `has_specific_target` | `[1, 1]` | int64 | 0 = use direction vectors, 1 = use waypoints below |
| `specific_target_positions` | `[1, 4, 3]` | float32 | 4 waypoint positions (x,y,z) in meters. Only used when `has_specific_target=1` |
| `specific_target_headings` | `[1, 4]` | float32 | 4 waypoint yaw angles in radians. Only used when `has_specific_target=1` |
| `allowed_pred_num_tokens` | `[1, 11]` | int64 | Binary mask for allowed prediction lengths. Index i allows (6+i) tokens = (6+i)*4 frames. All-ones = allow any length |

### Locomotion Modes

| Value | Name | Description |
|-------|------|-------------|
| 0 | IDLE | Standing still |
| 1 | SLOW_WALK | 0.1 - 0.8 m/s |
| 2 | WALK | 0.8 - 2.5 m/s |
| 3 | RUN | 2.5 - 7.5 m/s |
| 4 | IDLE_SQUAT | Squatting (height: 0.4 - 0.8 m) |
| 5 | IDLE_KNEEL_TWO_LEGS | Kneeling both knees (height: 0.2 - 0.4 m) |
| 6 | IDLE_KNEEL | Single-knee kneel (height: 0.2 - 0.4 m) |
| 7 | IDLE_LYING_FACE_DOWN | Lying face down |
| 8 | CRAWLING | Hand/knee crawl |
| 9 | IDLE_BOXING | Boxing stance (idle) |
| 10 | WALK_BOXING | Walking with boxing guard |
| 11 | LEFT_PUNCH | Left jab |
| 12 | RIGHT_PUNCH | Right jab |
| 13 | RANDOM_PUNCH | Random punch sequence |
| 14 | ELBOW_CRAWLING | Crawling on elbows |
| 15 | LEFT_HOOK | Left hook |
| 16 | RIGHT_HOOK | Right hook |
| 17 | FORWARD_JUMP | Jumping walk |
| 18 | STEALTH_WALK | Stealthy walk |
| 19 | INJURED_WALK | Limping walk |
| 20 | LEDGE_WALKING | Cautious ledge walk |
| 21 | OBJECT_CARRYING | Arms-out carrying walk |
| 22 | STEALTH_WALK_2 | Crouched stealth |
| 23 | HAPPY_DANCE_WALK | Dancing walk |
| 24 | ZOMBIE_WALK | Zombie walk |
| 25 | GUN_WALK | Gun stance walk |
| 26 | SCARE_WALK | Scared walk |

### Outputs (2 tensors)

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `mujoco_qpos` | `[1, 64, 36]` | float32 | Predicted trajectory at 30 Hz. Same 36-DOF format as input. **Padded** — must slice with `num_pred_frames` |
| `num_pred_frames` | `[1]` | int32 | Number of valid frames (24 - 64) |

### Python Example

```python
import onnxruntime as ort
import numpy as np

planner = ort.InferenceSession(
    "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    providers=["CPUExecutionProvider"],  # or "CUDAExecutionProvider"
)

# Build a default standing context (4 identical frames)
standing_qpos = np.zeros(36, dtype=np.float32)
standing_qpos[2] = 0.789       # root Z height (standing)
standing_qpos[3] = 1.0         # quaternion w=1 (identity rotation)
# Joint defaults in MuJoCo order:
default_joints_mujoco = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
     0.0, 0.0, 0.0,                           # waist
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # left arm
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)
standing_qpos[7:36] = default_joints_mujoco

context = np.tile(standing_qpos, (4, 1))[np.newaxis]  # [1, 4, 36]

inputs = {
    "context_mujoco_qpos": context,
    "mode":                 np.array([2], dtype=np.int64),                    # WALK
    "target_vel":           np.array([-1.0], dtype=np.float32),               # default speed
    "movement_direction":   np.array([[1.0, 0.0, 0.0]], dtype=np.float32),    # forward
    "facing_direction":     np.array([[1.0, 0.0, 0.0]], dtype=np.float32),    # face forward
    "height":               np.array([-1.0], dtype=np.float32),               # default
    "random_seed":          np.array([42], dtype=np.int64),
    "has_specific_target":  np.array([[0]], dtype=np.int64),
    "specific_target_positions": np.zeros([1, 4, 3], dtype=np.float32),
    "specific_target_headings":  np.zeros([1, 4], dtype=np.float32),
    "allowed_pred_num_tokens":   np.ones([1, 11], dtype=np.int64),
}

qpos_out, n_frames = planner.run(None, inputs)
n = int(n_frames[0])
trajectory = qpos_out[0, :n, :]  # [N, 36] valid frames at 30 Hz

# Parse each frame
for i, frame in enumerate(trajectory):
    root_pos  = frame[0:3]    # (x, y, z) meters
    root_quat = frame[3:7]    # (w, x, y, z) quaternion
    joints    = frame[7:36]   # 29 joint angles in radians (MuJoCo order)
    print(f"Frame {i}: pos={root_pos}, quat={root_quat}")
```

---

## 2. Encoder — `model_encoder.onnx`

**Path:** `gear_sonic_deploy/policy/release/model_encoder.onnx`

Compresses motion reference data (from the planner) and optional VR/SMPL inputs
into a compact 64-dimensional latent token. The encoder supports 3 modes; only
the relevant observations are filled, the rest are zero-padded.

### Input

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `obs_dict` | `[1, 1762]` | float32 | Flat concatenation of all encoder observations (see layout below) |

### Observation Layout (1762 dimensions)

| Offset | Dim | Name | Physical Meaning |
|--------|-----|------|------------------|
| 0 | 4 | `encoder_mode_4` | Encoder mode indicator. `[mode_id, 0, 0, 0]` where mode_id: 0=g1, 1=teleop, 2=smpl |
| 4 | 290 | `motion_joint_positions_10frame_step5` | 10 frames (step=5) x 29 joint angles (rad) from planner trajectory. Flattened as `[frame0_joint0, frame0_joint1, ..., frame9_joint28]` |
| 294 | 290 | `motion_joint_velocities_10frame_step5` | 10 frames (step=5) x 29 joint velocities (rad/s). Same layout |
| 584 | 10 | `motion_root_z_position_10frame_step5` | 10 frames x 1 root Z height (meters) |
| 594 | 1 | `motion_root_z_position` | Current root Z height (meters) |
| 595 | 6 | `motion_anchor_orientation` | Anchor orientation as 6D rotation (first 2 columns of 3x3 rotation matrix, flattened) |
| 601 | 60 | `motion_anchor_orientation_10frame_step5` | 10 frames x 6D rotation |
| 661 | 120 | `motion_joint_positions_lowerbody_10frame_step5` | 10 frames x 12 lower-body joint angles (rad) |
| 781 | 120 | `motion_joint_velocities_lowerbody_10frame_step5` | 10 frames x 12 lower-body joint velocities (rad/s) |
| 901 | 9 | `vr_3point_local_target` | 3 VR tracking points (left wrist, right wrist, head) x 3D position (meters) |
| 910 | 12 | `vr_3point_local_orn_target` | 3 VR tracking points x quaternion (w,x,y,z) |
| 922 | 720 | `smpl_joints_10frame_step1` | 10 frames x 24 SMPL joints x 3D position (x, y, z) in meters |
| 1642 | 60 | `smpl_anchor_orientation_10frame_step1` | 10 frames x 6D rotation |
| 1702 | 60 | `motion_joint_positions_wrists_10frame_step1` | 10 frames x 6 wrist joint angles (rad) |
| **1762** | | | **Total** |

### Encoder Modes

Only certain observation slots are active per mode. The rest should be **zero**:

- **Mode 0 (g1):** `encoder_mode_4`, `motion_joint_positions_10frame_step5`, `motion_joint_velocities_10frame_step5`, `motion_anchor_orientation_10frame_step5`
- **Mode 1 (teleop):** `encoder_mode_4`, `motion_joint_positions_lowerbody_10frame_step5`, `motion_joint_velocities_lowerbody_10frame_step5`, `vr_3point_local_target`, `vr_3point_local_orn_target`, `motion_anchor_orientation`
- **Mode 2 (smpl):** `encoder_mode_4`, `smpl_joints_10frame_step1`, `smpl_anchor_orientation_10frame_step1`, `motion_joint_positions_wrists_10frame_step1`

### Output

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `encoded_tokens` | `[1, 64]` | float32 | Latent motion embedding. Fed as `token_state` into the policy |

### Operational Modes — What Each Mode Requires

The system has 4 high-level operational modes. Each uses one encoder mode and fills
different subsets of the 1762-dim encoder input. All other slots are **zero**.

#### PLANNER mode (Encoder mode 0 — G1)

Locomotion planner active; full body (including upper body) controlled by planner.
Joysticks control direction and heading.

```
Data source: Planner trajectory only
Planner:     ACTIVE (provides full-body trajectory)
```

| Offset | Dim | Name | Source | Required |
|--------|-----|------|--------|----------|
| 0 | 4 | `encoder_mode_4` | `[0, 0, 0, 0]` | YES |
| 4 | 290 | `motion_joint_positions_10frame_step5` | Planner trajectory: 10 frames x 29 joints (full body) | YES |
| 294 | 290 | `motion_joint_velocities_10frame_step5` | Finite diff of planner joint positions | YES |
| 601 | 60 | `motion_anchor_orientation_10frame_step5` | Planner trajectory: 10 frames x 6D rotation | YES |
| all others | | | | ZERO |

#### PLANNER_FROZEN_UPPER mode (Encoder mode 0 — G1)

Planner locomotion; upper body frozen at last POSE snapshot. Same encoder mode as
PLANNER — the "frozen" behavior is handled externally by replaying a saved upper-body
pose into the planner trajectory before feeding it to the encoder.

```
Data source: Planner trajectory (lower body) + frozen snapshot (upper body)
Planner:     ACTIVE (lower body locomotion)
```

| Offset | Dim | Name | Source | Required |
|--------|-----|------|--------|----------|
| 0 | 4 | `encoder_mode_4` | `[0, 0, 0, 0]` | YES |
| 4 | 290 | `motion_joint_positions_10frame_step5` | Planner (legs) + frozen snapshot (upper body), all 29 joints | YES |
| 294 | 290 | `motion_joint_velocities_10frame_step5` | Finite diff (upper body ~0 since frozen) | YES |
| 601 | 60 | `motion_anchor_orientation_10frame_step5` | Planner trajectory: 10 frames x 6D rotation | YES |
| all others | | | | ZERO |

#### VR-3PT / TELEOP mode (Encoder mode 1 — Teleop)

Planner locomotion for lower body; upper body follows VR 3-point tracking
(head + left hand + right hand). Depends on non-IK-based VR 3-point calibration.

```
Data source: Planner (lower body) + VR headset (upper body 3-point targets)
Planner:     ACTIVE (lower body locomotion only)
VR input:    3 tracking points — left wrist, right wrist, torso/head
```

| Offset | Dim | Name | Source | Required |
|--------|-----|------|--------|----------|
| 0 | 4 | `encoder_mode_4` | `[1, 0, 0, 0]` | YES |
| 595 | 6 | `motion_anchor_orientation` | Current 6D rotation of anchor frame (from planner or robot) | YES |
| 661 | 120 | `motion_joint_positions_lowerbody_10frame_step5` | Planner trajectory: 10 frames x 12 lower-body joints | YES |
| 781 | 120 | `motion_joint_velocities_lowerbody_10frame_step5` | Finite diff of lower-body joint positions | YES |
| 901 | 9 | `vr_3point_local_target` | VR tracking: `[L_wrist_xyz, R_wrist_xyz, head_xyz]` root-relative (m) | YES |
| 910 | 12 | `vr_3point_local_orn_target` | VR tracking: `[L_wrist_wxyz, R_wrist_wxyz, head_wxyz]` quaternions | YES |
| all others | | | | ZERO |

The 3 VR tracking points (from `vr_3point_index = [28, 29, 9]` in IsaacLab order):

| Point | Body Part | Position (3D) | Orientation (quat) |
|-------|-----------|---------------|-------------------|
| 0 | Left wrist | `vr_3point_local_target[0:3]` | `vr_3point_local_orn_target[0:4]` |
| 1 | Right wrist | `vr_3point_local_target[3:6]` | `vr_3point_local_orn_target[4:8]` |
| 2 | Torso / head | `vr_3point_local_target[6:9]` | `vr_3point_local_orn_target[8:12]` |

All positions are **root-relative** (normalized to the robot's current root frame).

#### POSE mode (Encoder mode 2 — SMPL)

Whole-body teleop — streaming the SMPL pose from PICO VR to the deployment side.
Your motion directly maps to the robot. No planner needed for motion generation;
the SMPL body model provides the full reference.

```
Data source: SMPL body model (from PICO VR / video / mocap)
Planner:     NOT NEEDED for motion (may still run for context)
```

| Offset | Dim | Name | Source | Required |
|--------|-----|------|--------|----------|
| 0 | 4 | `encoder_mode_4` | `[2, 0, 0, 0]` | YES |
| 922 | 720 | `smpl_joints_10frame_step1` | SMPL model: 10 consecutive frames x 24 joints x 3D position (x, y, z) in meters | YES |
| 1642 | 60 | `smpl_anchor_orientation_10frame_step1` | SMPL model: 10 frames x 6D rotation of root/anchor | YES |
| 1702 | 60 | `motion_joint_positions_wrists_10frame_step1` | Wrist joints: 10 frames x 6 wrist angles (rad) from SMPL retargeting | YES |
| all others | | | | ZERO |

The 24 SMPL joints follow the standard SMPL skeleton order (Pelvis, L_Hip, R_Hip,
Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle, Spine3, L_Foot, R_Foot, Neck,
Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist, L_Hand, R_Hand,
L_Fingers, R_Fingers). Each joint has 3 coordinates (x, y, z) in meters.

### Summary Table

| Mode | Encoder | Planner | Upper Body Source | Encoder Obs Filled (dims) | Encoder Obs Zeroed (dims) |
|------|---------|---------|-------------------|--------------------------|--------------------------|
| PLANNER | G1 (0) | Active (full body) | Planner | 644 (4+290+290+60) | 1118 |
| PLANNER_FROZEN_UPPER | G1 (0) | Active (lower body) | Frozen snapshot | 644 (4+290+290+60) | 1118 |
| VR-3PT / TELEOP | Teleop (1) | Active (lower body) | VR 3-point tracking | 271 (4+6+120+120+9+12) | 1491 |
| POSE | SMPL (2) | Not needed | SMPL from PICO VR | 844 (4+720+60+60) | 918 |

### Python Example

```python
encoder = ort.InferenceSession(
    "gear_sonic_deploy/policy/release/model_encoder.onnx",
    providers=["CPUExecutionProvider"],
)

encoder_obs = np.zeros([1, 1762], dtype=np.float32)

# --- Mode 0 (g1): fill from planner trajectory ---
encoder_obs[0, 0] = 0.0  # mode_id = 0 (g1)

# Extract 10 frames from planner trajectory at step=5 intervals
# (frames 0, 5, 10, 15, ..., 45 — or repeat last if not enough)
for i in range(10):
    frame_idx = min(i * 5, n - 1)
    joints = trajectory[frame_idx, 7:36]  # 29 joints in MuJoCo order
    encoder_obs[0, 4 + i*29 : 4 + (i+1)*29] = joints  # joint positions

# Similarly fill velocities at offset 294, anchor orientations at offset 601, etc.
# (Velocity can be estimated via finite differences of joint positions)

tokens = encoder.run(None, {"obs_dict": encoder_obs})
token_state = tokens[0]  # [1, 64]
```

---

## 3. Policy / Decoder — `model_decoder.onnx`

**Path:** `gear_sonic_deploy/policy/release/model_decoder.onnx`

The main RL control policy. Takes the encoder's latent tokens plus 10 frames of
real robot state history and outputs 29 joint commands.

### Input

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `obs_dict` | `[1, 994]` | float32 | Flat concatenation of policy observations (see layout below) |

### Observation Layout (994 dimensions)

| Offset | Dim | Name | Physical Meaning | Units |
|--------|-----|------|------------------|-------|
| 0 | 64 | `token_state` | Encoder output (latent tokens) | — |
| 64 | 30 | `his_base_angular_velocity_10frame_step1` | 10 frames x 3 angular velocity from IMU | rad/s |
| 94 | 290 | `his_body_joint_positions_10frame_step1` | 10 frames x 29 current robot joint positions | rad |
| 384 | 290 | `his_body_joint_velocities_10frame_step1` | 10 frames x 29 current robot joint velocities | rad/s |
| 674 | 290 | `his_last_actions_10frame_step1` | 10 frames x 29 previous raw policy outputs | normalized |
| 964 | 30 | `his_gravity_dir_10frame_step1` | 10 frames x 3D gravity direction in body frame | unit vector |
| **994** | | | **Total** | |

Note: "10frame_step1" means 10 consecutive frames at the 50 Hz control rate.
Frame 0 is the oldest, frame 9 is the most recent.

### Output

| Tensor Name | Shape | Dtype | Physical Meaning |
|-------------|-------|-------|------------------|
| `action` | `[1, 29]` | float32 | Normalized joint actions in **IsaacLab order** |

### Action Post-Processing

The raw action output must be converted to joint position targets:

```python
q_target[mujoco_idx] = default_angle[mujoco_idx] + action[isaaclab_idx] * action_scale[mujoco_idx]
```

where `isaaclab_idx = mujoco_to_isaaclab[mujoco_idx]`.

### Constants (MuJoCo order, 29 joints)

```python
# Default standing angles (radians, MuJoCo order)
default_angles = np.array([
    -0.312,  0.0,    0.0,    0.669, -0.363,  0.0,    # left leg
    -0.312,  0.0,    0.0,    0.669, -0.363,  0.0,    # right leg
     0.0,    0.0,    0.0,                              # waist
     0.2,    0.2,    0.0,    0.6,   0.0,  0.0,  0.0,  # left arm
     0.2,   -0.2,    0.0,    0.6,   0.0,  0.0,  0.0,  # right arm
])

# Action scale (MuJoCo order) = 0.25 * effort_limit / stiffness
# Motor constants:
#   armature_5020 = 0.003609725,  effort_5020 = 25.0
#   armature_7520_14 = 0.010177520, effort_7520_14 = 88.0
#   armature_7520_22 = 0.025101925, effort_7520_22 = 139.0
#   armature_4010 = 0.00425,      effort_4010 = 5.0
# stiffness = armature * (10 * 2 * pi)^2
# action_scale = 0.25 * effort / stiffness
ONE_FREQ = 10 * 2 * np.pi
action_scale = np.array([
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # left_hip_pitch  (7520_22)
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # left_hip_roll   (7520_22)
    0.25 *  88.0 / (0.010177520 * ONE_FREQ**2),  # left_hip_yaw    (7520_14)
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # left_knee       (7520_22)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_ankle_pitch (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_ankle_roll  (5020)
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # right_hip_pitch (7520_22)
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # right_hip_roll  (7520_22)
    0.25 *  88.0 / (0.010177520 * ONE_FREQ**2),  # right_hip_yaw   (7520_14)
    0.25 * 139.0 / (0.025101925 * ONE_FREQ**2),  # right_knee      (7520_22)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_ankle_pitch (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_ankle_roll  (5020)
    0.25 *  88.0 / (0.010177520 * ONE_FREQ**2),  # waist_yaw       (7520_14)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # waist_roll      (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # waist_pitch     (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_shoulder_pitch  (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_shoulder_roll   (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_shoulder_yaw    (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_elbow      (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # left_wrist_roll (5020)
    0.25 *   5.0 / (0.00425     * ONE_FREQ**2),  # left_wrist_pitch (4010)
    0.25 *   5.0 / (0.00425     * ONE_FREQ**2),  # left_wrist_yaw   (4010)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_shoulder_pitch (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_shoulder_roll  (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_shoulder_yaw   (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_elbow     (5020)
    0.25 *  25.0 / (0.003609725 * ONE_FREQ**2),  # right_wrist_roll (5020)
    0.25 *   5.0 / (0.00425     * ONE_FREQ**2),  # right_wrist_pitch (4010)
    0.25 *   5.0 / (0.00425     * ONE_FREQ**2),  # right_wrist_yaw   (4010)
])

# PD gains (MuJoCo order) — for low-level motor control
# kp = stiffness (2x for ankle/waist joints)
# kd = damping   (2x for ankle/waist joints)
# stiffness = armature * omega^2,  damping = 2 * zeta * armature * omega  (zeta=2.0)
```

### Python Example

```python
policy = ort.InferenceSession(
    "gear_sonic_deploy/policy/release/model_decoder.onnx",
    providers=["CPUExecutionProvider"],
)

# Build observation vector (normally from a 50 Hz control loop)
policy_obs = np.zeros([1, 994], dtype=np.float32)

# [0:64]    token_state from encoder
policy_obs[0, 0:64] = token_state[0]

# [64:94]   10 frames of angular velocity (rad/s) — from IMU
# [94:384]  10 frames of 29 joint positions (rad) — from robot joint encoders
# [384:674] 10 frames of 29 joint velocities (rad/s) — from robot joint encoders
# [674:964] 10 frames of 29 previous actions — raw policy outputs (before scaling)
# [964:994] 10 frames of gravity direction in body frame — from IMU

# Example: fill with standing pose for all 10 frames
for i in range(10):
    policy_obs[0, 94 + i*29 : 94 + (i+1)*29] = default_angles  # joint positions
    policy_obs[0, 964 + i*3 : 964 + (i+1)*3] = [0.0, 0.0, -1.0]  # gravity = down

raw_action = policy.run(None, {"obs_dict": policy_obs})
action = raw_action[0][0]  # [29] in IsaacLab order

# Convert to joint position targets (MuJoCo order)
q_target = np.zeros(29)
for mj_idx in range(29):
    il_idx = mujoco_to_isaaclab[mj_idx]
    q_target[mj_idx] = default_angles[mj_idx] + action[il_idx] * action_scale[mj_idx]

print("Joint targets (rad, MuJoCo order):", q_target)
```

---

## Full Pipeline Example

```python
import onnxruntime as ort
import numpy as np

# ── Load models ──────────────────────────────────────────────────────────────
planner = ort.InferenceSession("gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
                                providers=["CPUExecutionProvider"])
encoder = ort.InferenceSession("gear_sonic_deploy/policy/release/model_encoder.onnx",
                                providers=["CPUExecutionProvider"])
policy  = ort.InferenceSession("gear_sonic_deploy/policy/release/model_decoder.onnx",
                                providers=["CPUExecutionProvider"])

# ── Joint ordering ───────────────────────────────────────────────────────────
mujoco_to_isaaclab = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18,
    25, 19, 26, 20, 27, 21, 28
])

default_angles = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

ONE_FREQ = 10 * 2 * np.pi
action_scale = np.array([
    0.25*139.0/(0.025101925*ONE_FREQ**2), 0.25*139.0/(0.025101925*ONE_FREQ**2),
    0.25*88.0/(0.010177520*ONE_FREQ**2),  0.25*139.0/(0.025101925*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*139.0/(0.025101925*ONE_FREQ**2), 0.25*139.0/(0.025101925*ONE_FREQ**2),
    0.25*88.0/(0.010177520*ONE_FREQ**2),  0.25*139.0/(0.025101925*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*88.0/(0.010177520*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*5.0/(0.00425*ONE_FREQ**2),       0.25*5.0/(0.00425*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*25.0/(0.003609725*ONE_FREQ**2),
    0.25*25.0/(0.003609725*ONE_FREQ**2),  0.25*5.0/(0.00425*ONE_FREQ**2),
    0.25*5.0/(0.00425*ONE_FREQ**2),
])

# ── Step 1: Planner — generate walk-forward trajectory ───────────────────────
standing_qpos = np.zeros(36, dtype=np.float32)
standing_qpos[2] = 0.789
standing_qpos[3] = 1.0
standing_qpos[7:36] = default_angles.astype(np.float32)
context = np.tile(standing_qpos, (4, 1))[np.newaxis]  # [1, 4, 36]

planner_out = planner.run(None, {
    "context_mujoco_qpos":       context,
    "mode":                      np.array([2], dtype=np.int64),
    "target_vel":                np.array([-1.0], dtype=np.float32),
    "movement_direction":        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    "facing_direction":          np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    "height":                    np.array([-1.0], dtype=np.float32),
    "random_seed":               np.array([42], dtype=np.int64),
    "has_specific_target":       np.array([[0]], dtype=np.int64),
    "specific_target_positions": np.zeros([1, 4, 3], dtype=np.float32),
    "specific_target_headings":  np.zeros([1, 4], dtype=np.float32),
    "allowed_pred_num_tokens":   np.ones([1, 11], dtype=np.int64),
})
traj = planner_out[0][0, :int(planner_out[1][0]), :]  # [N, 36]
print(f"Planner generated {len(traj)} frames at 30 Hz")

# ── Step 2: Encoder — compress trajectory into tokens ────────────────────────
enc_obs = np.zeros([1, 1762], dtype=np.float32)
enc_obs[0, 0] = 0.0  # g1 mode

# Fill 10 frames of joint positions at step=5 intervals
for i in range(10):
    fidx = min(i * 5, len(traj) - 1)
    enc_obs[0, 4 + i*29 : 4 + (i+1)*29] = traj[fidx, 7:36]

# Fill 10 frames of anchor orientation at offset 601 (6D rotation per frame)
# (simplified — in production, compute from planner quaternion output)

token_state = encoder.run(None, {"obs_dict": enc_obs})[0]  # [1, 64]

# ── Step 3: Policy — generate motor commands ─────────────────────────────────
pol_obs = np.zeros([1, 994], dtype=np.float32)
pol_obs[0, 0:64] = token_state[0]

# Fill 10 frames of robot state (using standing defaults as placeholder)
for i in range(10):
    pol_obs[0, 94  + i*29 : 94  + (i+1)*29] = default_angles   # joint pos
    pol_obs[0, 964 + i*3  : 964 + (i+1)*3]  = [0.0, 0.0, -1.0] # gravity

raw_action = policy.run(None, {"obs_dict": pol_obs})[0][0]  # [29] IsaacLab order

# Convert to joint targets
q_target = np.zeros(29)
for mj in range(29):
    q_target[mj] = default_angles[mj] + raw_action[mujoco_to_isaaclab[mj]] * action_scale[mj]

print("Joint targets (rad):", q_target)
```

---

## Notes

- **Coordinate frame:** MuJoCo Z-up (X=forward, Y=left, Z=up). The C++ deployment
  normalizes the planner context to a canonical zero-yaw frame before inference and
  rotates the output back. If you use the planner standalone, be aware of this.
- **History buffers:** The encoder and policy expect rolling history windows. In a
  real control loop, maintain ring buffers of past observations and shift them each
  tick.
- **Units:** All angles in radians, positions in meters, velocities in rad/s or m/s.
- **Frame naming:** `10frame_step5` = 10 samples spaced 5 frames apart in the
  planner's 30 Hz output. `10frame_step1` = 10 consecutive samples at 50 Hz
  control rate.
- **ONNX Runtime providers:** Use `"CUDAExecutionProvider"` for GPU acceleration,
  or `"TensorrtExecutionProvider"` for TensorRT (matches the C++ deployment path).

---

## Appendix: PICO VR Sender — How `body_quat` Is Computed from SMPL

**Source:** `gear_sonic/scripts/pico_manager_thread_server.py`

In POSE mode (encoder mode 2), the ZMQ protocol requires a `body_quat` field alongside
`smpl_joints` and `smpl_pose`. This quaternion represents the SMPL root (pelvis)
orientation in the robot's Z-up coordinate frame. The PICO sender computes it as follows.

### Pipeline

```
PICO VR headset (XRoboToolkit SDK)
  │
  │  xrt.get_body_joints_pose()
  │  → 24 joints × [x, y, z, qx, qy, qz, qw]
  │
  ▼
compute_from_body_poses()          ← convert PICO body tracking to SMPL format
  │
  │  1. Reorder quaternion components to wxyz
  │  2. Apply 180° Y-axis flip
  │  3. Convert global rotations to parent-relative (local) rotations
  │  4. Separate root (joint 0) from body (joints 1–23)
  │     → global_orient: root axis-angle (3,)
  │     → body_pose: 23 joints × 3 axis-angle = (69,)
  │
  ▼
process_smpl_joints()              ← apply coordinate transforms, run SMPL FK
  │
  │  Step 1: angle_axis_to_quaternion(global_orient)
  │           Convert root axis-angle to quaternion
  │
  │  Step 2: smpl_root_ytoz_up(quat)
  │           Rotate 90° about X-axis (SMPL Y-up → robot Z-up)
  │           Formula: Rot_x(π/2) × quat
  │
  │  Step 3: compute_human_joints(body_pose, global_orient_new)
  │           SMPL forward kinematics → 24 joint positions in world frame
  │
  │  Step 4: remove_smpl_base_rot(quat)
  │           Remove SMPL rest-pose offset (120° rotation about [1,1,1])
  │           Formula: quat × conjugate([0.5, 0.5, 0.5, 0.5])
  │
  │  Returns: global_orient_quat  ← this becomes body_quat
  │
  ▼
PicoPoseStreamer.run_once()         ← interpolate, buffer, and send
  │
  │  1. Extract: body_quat = latest_data["global_orient_quat"]
  │  2. Interpolate: quat_lerp(prev_body_quat, body_quat, alpha)
  │  3. Buffer N frames
  │  4. Pack into ZMQ message as "body_quat_w" field, shape [N, 4]
  │
  ▼
ZMQ send → gear_sonic_deploy receiver
```

### Coordinate Transforms (in `gear_sonic/isaac_utils/rotations.py`)

**`smpl_root_ytoz_up` (line 712):**
SMPL uses Y-up; the robot uses Z-up. A 90° rotation about the X-axis maps between them:

```python
@torch.jit.script
def smpl_root_ytoz_up(root_quat_y_up) -> torch.Tensor:
    base_rot = angle_axis_to_quaternion(
        torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
    )
    return quat_mul(base_rot, root_quat_y_up, w_last=False)
```

**`remove_smpl_base_rot` (line 704):**
SMPL's rest pose has a built-in 120° rotation about the `[1,1,1]` axis (quaternion
`[0.5, 0.5, 0.5, 0.5]`). This must be conjugated out to align with a neutral standing
pose in robot coordinates:

```python
@torch.jit.script
def remove_smpl_base_rot(quat, w_last: bool):
    base_rot = quat_conjugate(
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat), w_last=w_last
    )
    return quat_mul(quat, base_rot, w_last=w_last)
```

### How `body_quat` Is Used on the Receiver Side

On the deployment side (`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`),
`body_quat` is stored as `BodyQuaternions(frame)[0]` in the `MotionSequence` and is used
exclusively by `GatherMotionAnchorOrientationMutiFrame()` (line 514) to compute the
`smpl_anchor_orientation` encoder observation:

1. **Heading alignment:** Computes a yaw-only correction that maps the motion's initial
   facing direction to the robot's initial facing direction (plus optional joystick delta).
2. **Relative rotation:** `conjugate(robot_base_quat) × heading_corrected_ref_quat` —
   expresses "how far the robot must rotate to match the reference" in the robot's local
   frame.
3. **6D encoding:** Extracts the first 2 columns of the resulting 3×3 rotation matrix
   (6 values per frame, 10 frames = 60 dimensions total).

This means `body_quat` is **not** passed directly to the encoder. It is transformed into
a robot-relative 6D rotation representation before being fed as `smpl_anchor_orientation`.

---

## Appendix: SMPL Forward Kinematics in the PICO Sender

**Source:** `gear_sonic/trl/utils/torch_transform.py` — `compute_human_joints()`

The `process_smpl_joints()` function calls `compute_human_joints()` to run a lightweight
joints-only SMPL forward kinematics pass. This computes the 24 world-space joint positions
that become `smpl_joints` in the ZMQ message.

### Inputs

- **`body_pose`** (63,) — 21 local joint rotations as axis-angle (joints 1–21, 3 each)
- **`global_orient`** (3,) — root pelvis rotation as axis-angle
- **`human_joints_info.pkl`** — precomputed rest-pose data:
  - `J` (55, 3) — rest-pose (T-pose) joint positions
  - `parents_list` — parent index for each of the 55 joints

### Algorithm

**1. Build full pose and convert to rotation matrices**

```python
# Pad to 55 joints: [global_orient(3), body_pose(63), zeros(99)]
full_pose = torch.cat([global_orient, body_pose, zeros(99)], dim=-1)  # (165,)
rot_mats = angle_axis_to_rotation_matrix(full_pose.reshape(55, 3))    # (55, 3, 3)
```

The 99 zeros pad out to 55 total joints (SMPL has extra hand/face joints beyond the
22 body joints; these remain in rest pose).

**2. Compute bone vectors (relative joint offsets)**

```python
rel_joints = J.clone()
rel_joints[..., 1:, :] -= J[..., parents_list[1:], :]  # child_pos - parent_pos
```

Subtracts each parent's rest position from the child's, giving the translation from
parent to child in the rest pose (the "bone length and direction").

**3. Build 4x4 homogeneous transforms per joint**

```python
transforms_mat = [rot_mats | rel_joints]   # upper 3x4
                  [  0 0 0  |     1     ]   # bottom row
```

Each joint's local transform combines its local rotation with the bone vector to it
from its parent.

**4. Chain transforms down the skeleton tree**

```python
transform_chain[0] = transforms_mat[0]                              # root (world frame)
for i in range(1, 55):
    transform_chain[i] = transform_chain[parents_list[i]] @ transforms_mat[i]
```

This is the core FK: each joint's world transform = parent's world transform x local
transform. Walking from root to leaves, each matrix multiplication accumulates the
rotations and translations down the kinematic chain.

**5. Extract world-space joint positions**

```python
joints = torch.stack(transform_chain)[..., :3, 3]  # translation column of each 4x4
```

**6. Select output joints**

```python
output_joint_index = [0..21] + [39, 54]  # 22 body joints + 2 thumb tips = 24 joints
```

### Data flow diagram

```
Rest-pose skeleton             Local rotations (from PICO)
  J = (55, 3)                  rot_mats = (55, 3, 3)
       │                              │
       ▼                              ▼
 bone vectors                  4x4 transform per joint
 child - parent                [ R  | bone_vec ]
       │                       [ 0  |    1     ]
       │                              │
       └──────────┬───────────────────┘
                  ▼
     chain: T_world[i] = T_world[parent[i]] x T_local[i]
                  │
                  ▼
     extract [:3, 3] from each 4x4  →  world positions (55, 3)
                  │
                  ▼
     select 22 body + 2 thumb tips  →  (24, 3)
```

### Notes

- This is a **skeleton-only** FK pass — no mesh vertices, no shape blending (betas).
  The rest-pose joint positions in `human_joints_info.pkl` are pre-baked for a fixed
  body shape.
- The 24 output joints are then made root-relative in `process_smpl_joints()` by
  rotating them with `quat_inv(global_orient_quat)`, producing `smpl_joints_local`
  which is sent as `smpl_joints` over ZMQ.

---

## Appendix: End-to-End SMPL Data Pipeline (XRT → SONIC Encoder)

This section traces every transformation applied to SMPL data from the PICO VR headset
all the way to the SONIC encoder input. The pipeline spans two processes: the Python
sender (`gear_sonic/scripts/pico_manager_thread_server.py`) and the C++ deployment
(`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`).

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PICO VR HEADSET                                                         │
│                                                                         │
│  xrt.get_body_joints_pose()                                            │
│  → 24 joints × [x, y, z, qx, qy, qz, qw]  (Unity frame, scalar-last)│
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────────────┐
          ▼                    ▼                              ▼
   ┌─────────────┐   ┌─────────────────┐          ┌──────────────────┐
   │ smpl_joints  │   │ body_quat       │          │ joint_pos        │
   │ (24, 3)      │   │ (4,)            │          │ wrist angles (6) │
   └──────┬──────┘   └───────┬─────────┘          └────────┬─────────┘
          │                   │                              │
          ▼                   ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ZMQ MESSAGE  (protocol v2 or v3)                                        │
│  fields: smpl_joints, smpl_pose, body_quat_w, joint_pos, frame_index…  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ C++ RECEIVER  (gear_sonic_deploy)                                       │
│  ZMQ decode → StreamedMotionMerger → MotionSequence                    │
│  SetEncodeMode(2) for protocol v2/v3                                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────────────┐
          ▼                    ▼                              ▼
   smpl_joints_          body_quaternions_           joint_positions_
   10frame_step1         10frame_step1               wrists_10frame_step1
   (720 dims)            (used for anchor ori)       (60 dims)
          │                    │                              │
          │                    ▼                              │
          │            smpl_anchor_orientation_               │
          │            10frame_step1                          │
          │            (60 dims)                              │
          ▼                    ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ENCODER INPUT  (mode 2 = SMPL)                                          │
│                                                                         │
│  [encoder_mode_4] [smpl_joints_10f] [smpl_anchor_ori_10f] [wrists_10f] │
│    4 dims            720 dims           60 dims             60 dims     │
│                                                        total = 844 dims │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
                      ONNX encoder → 64-dim token
```

### Stage 1: PICO XRoboToolkit → Raw Body Poses

**File:** `pico_manager_thread_server.py`, `PicoReaderThread` (line 787)

```python
body_poses = xrt.get_body_joints_pose()   # 24 joints × 7 floats
sample = {"body_poses_np": np.array(body_poses), ...}
```

The PICO headset runs its own body tracking model and outputs 24 SMPL joint poses in
Unity convention: `[x, y, z, qx, qy, qz, qw]` per joint. This is polled in a
background thread and stored as the latest sample.

### Stage 2: Global-to-Local Rotation Conversion

**File:** `pico_manager_thread_server.py`, `compute_from_body_poses()` (line 560)

```python
global_quats = body_poses_np[:, [6, 3, 4, 5]]          # reorder to wxyz
global_rots = sRot.from_quat(global_quats, scalar_first=True)
global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

for i in range(24):
    if parent_indices[i] == -1:
        local_rots[i] = global_rots[i]                  # root: local = global
    else:
        local_rots[i] = parent.inv() * global_rots[i]   # parent-relative

global_orient = pose_aa[0]     # root axis-angle (3,)
body_pose     = pose_aa[1:]    # joints 1-23 axis-angle (69,)
```

PICO provides **global** (world-frame) rotations. SMPL expects **local**
(parent-relative) rotations. The conversion is `local = parent_global⁻¹ × child_global`.
A 180° Y-axis flip aligns PICO's facing convention with SMPL's.

### Stage 3: SMPL Forward Kinematics + Coordinate Transforms

**File:** `pico_manager_thread_server.py`, `process_smpl_joints()` (line 449)

This function produces three outputs from the SMPL parameters:

**Output 1: `smpl_joints_local` (24, 3) → sent as `smpl_joints`**

```
global_orient  →  angle_axis_to_quaternion
                        │
                  smpl_root_ytoz_up          (90° X rotation: Y-up → Z-up)
                        │
                  quaternion_to_angle_axis   → global_orient_new
                        │
              compute_human_joints(body_pose, global_orient_new)
                        │
                  joints (24, 3) in world frame
                        │
              quat_apply(quat_inv(global_orient_quat), joints)
                        │
                  smpl_joints_local (24, 3)  — root-relative positions in meters
```

The joints are rotated by the inverse root orientation to make them **root-relative**
(invariant to which direction the person is facing).

**Output 2: `global_orient_quat` (4,) → sent as `body_quat_w`**

```
global_orient  →  angle_axis_to_quaternion
                        │
                  smpl_root_ytoz_up          (Y-up → Z-up)
                        │
                  remove_smpl_base_rot       (remove 120° [1,1,1] rest offset)
                        │
                  global_orient_quat (4,)    — pelvis orientation in robot Z-up frame
```

**Output 3: `smpl_pose` (21, 3) — passed through unchanged**

The 21 local joint rotations (axis-angle) from joints 1–21 are sent directly.
(Joint 0 = root is separate; joints 22–23 = hands are excluded.)

### Stage 4: Wrist Retargeting (SMPL → G1 Robot Joints)

**File:** `pico_manager_thread_server.py`, `run_once()` (lines 1345–1403)

The SMPL elbow and wrist axis-angle rotations are retargeted to G1 robot wrist joints:

```python
# Decompose SMPL elbow rotation into twist (along Y) and swing components
g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, [0,1,0])

# Convert swing to Euler, combine with SMPL wrist Euler
g1_l_wrist_roll  = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
g1_l_wrist_pitch = -l_wrist_euler[:, 1]
g1_l_wrist_yaw   = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

# Write into joint_pos at G1 wrist indices (23–28)
joint_pos[G1_L_WRIST_ROLL_IDX]  = g1_l_wrist_roll
joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch
joint_pos[G1_L_WRIST_YAW_IDX]   = g1_l_wrist_yaw
# (same for right wrist)
```

The G1 robot has 3-DOF wrists (roll/pitch/yaw) but SMPL has separate elbow and wrist
joints. The elbow's "swing" component (non-twist rotation that the G1 elbow can't
reproduce) is folded into the wrist angles. Only 6 of the 29 `joint_pos` values are
non-zero; the rest are zero-filled.

### Stage 5: Interpolation and Buffering

**File:** `pico_manager_thread_server.py`, `run_once()` (lines 1330–1341)

```python
alpha = (target_ns - prev_stamp_ns) / (curr_stamp_ns - prev_stamp_ns)

use_joints    = (1 - alpha) * prev_smpl_joints + alpha * smpl_joints      # linear
use_pose      = _interp_pose_axis_angle(prev_pose, pose, alpha)            # slerp via quaternions
use_body_quat = _quat_lerp_normalized(prev_body_quat, body_quat, alpha)   # nlerp
```

PICO data arrives at variable rate. The sender resamples to a fixed `target_fps` using
interpolation between consecutive PICO frames, then buffers N frames before sending.

### Stage 6: ZMQ Packing and Sending

**File:** `pico_manager_thread_server.py` (lines 1438–1468),
`gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`

```python
numpy_data = {
    "smpl_joints":   np.stack(frame_buffer["smpl_joints"]),    # (N, 24, 3) float32
    "smpl_pose":     np.stack(frame_buffer["smpl_pose"]),      # (N, 21, 3) float32
    "body_quat_w":   np.stack(frame_buffer["body_quat_w"]),    # (N, 4) float32
    "joint_pos":     np.stack(frame_buffer["joint_pos"]),       # (N, 29) float64
    "joint_vel":     np.zeros((N, 29)),                         # (N, 29) zeros
    "frame_index":   np.array(frame_buffer["frame_index"]),     # (N,) int64
    "heading_increment": np.array([yaw_delta]),                  # (1,) float32
    ...
}
packed_message = pack_pose_message(numpy_data, topic="pose")
socket.send(packed_message)
```

The ZMQ wire format is: `[topic bytes][1280-byte JSON header][binary payload]`.
The JSON header describes each field's name, dtype, and shape.

### Stage 7: C++ ZMQ Decode

**File:** `zmq_endpoint_interface.hpp`, `DecodeIntoMotionSequence()` (line 556)

The C++ receiver:
1. Parses the JSON header to locate fields by name
2. Detects protocol version (v2 if `smpl_joints` + `smpl_pose` required)
3. Decodes binary buffers into typed C++ vectors (memcpy + optional byte-swap)
4. **No value transformation** — raw floats are cast to doubles

The `body_quat_w` field is decoded as `decoded_body_quat[frame][body][wxyz]`.

### Stage 8: StreamedMotionMerger → MotionSequence

**File:** `streamed_motion_merger.hpp`, `MergeIncomingData()` (line 125)

Incoming frames are merged into a sliding-window `MotionSequence` using `frame_index`
for alignment. Data is **copied without transformation**:

```cpp
// SMPL joints: direct copy
motion->SmplJoints(dst)[joint][xyz] = data.smpl_joints[frame][joint][xyz];

// Body quaternions: direct copy
motion->BodyQuaternions(dst)[body][wxyz] = data.body_quat[frame][body][wxyz];
```

For protocol v2/v3, the encode mode is set:
```cpp
merge_result.motion->SetEncodeMode(2);  // SMPL-based encoder
```

### Stage 9: Encoder Observation Gathering

**File:** `g1_deploy_onnx_ref.cpp` (lines 777–894, 514–602)

The encoder runs at 50 Hz. Each tick, it gathers observations for encoder mode 2 (SMPL)
as configured in `observation_config.yaml` (lines 74–80):

| Observation | Dims | Gatherer | What it does |
|---|---|---|---|
| `encoder_mode_4` | 4 | `GatherEncoderMode(3)` | `[2, 0, 0, 0]` — one-hot-ish mode ID |
| `smpl_joints_10frame_step1` | 720 | `GatherMotionSmplJointsMultiFrame(10, 1)` | **Direct copy** of 10 consecutive frames × 24 joints × 3 coords from `MotionSequence::smpl_joints_`. No scaling. Values in meters, root-relative. |
| `smpl_anchor_orientation_10frame_step1` | 60 | `GatherMotionAnchorOrientationMutiFrame(10, 1)` | **Computed** from `MotionSequence::body_quaternions_`. For each of 10 frames: heading-correct the motion's root quat, compute relative rotation to robot's current base, convert to 6D (first 2 cols of 3×3 matrix). |
| `motion_joint_positions_wrists_10frame_step1` | 60 | `GatherMotionJointPositionsMultiFrame(10, 1, wrist_indices)` | **Direct copy** of 10 frames × 6 wrist joint angles (radians) from `MotionSequence::joint_positions_`. These are the retargeted G1 wrist values from Stage 4. |

### Stage 10: ONNX Encoder Inference

The gathered 844-dimensional observation vector is fed to `model_encoder.onnx`:

```
Input:  obs_dict [1, 844]
Output: encoded_tokens [1, 64]
```

The 64-dim token is then concatenated with robot state history (joint positions,
velocities, actions, angular velocity, gravity — 930 dims) and fed to the policy
decoder to produce 29 motor commands.

### Summary of Transformations

| Stage | Data | Transform |
|---|---|---|
| 1. XRT raw | 24 × [x,y,z,qx,qy,qz,qw] | None (raw PICO output) |
| 2. Global→local | 24 quaternions → 24 axis-angle | `parent⁻¹ × child` + 180° Y flip |
| 3a. FK joints | body_pose + global_orient → 24 joint positions | SMPL FK (chain 4×4 transforms) |
| 3b. Root-relative | world joints → local joints | `quat_inv(root) × joints` |
| 3c. body_quat | global_orient → pelvis quat | Y→Z up (90° X) + remove SMPL base rot |
| 4. Wrist retarget | SMPL elbow/wrist → G1 wrist angles | Twist-swing decomposition + Euler combine |
| 5. Interpolation | Variable rate → fixed fps | Linear (joints), slerp (poses), nlerp (quat) |
| 6. ZMQ pack | numpy arrays → binary | float32/64 serialization (no value change) |
| 7. C++ decode | binary → C++ vectors | memcpy + float→double cast |
| 8. Merge | frames → sliding window | Direct copy (no value change) |
| 9a. smpl_joints | MotionSequence → encoder buffer | **Direct copy** (no scaling) |
| 9b. anchor_ori | body_quat → 6D rotation | Heading correct + relative rotation + mat[:, :2] |
| 9c. wrist joints | MotionSequence → encoder buffer | **Direct copy** (no scaling) |
| 10. Encoder | 844-dim obs → 64-dim token | ONNX neural network inference |

---

## Appendix: Decoder Output and PD Control

The policy decoder does **not** output motor torques or motion trajectories. It outputs
**29 dimensionless action values** that are converted to PD position targets for the
robot's onboard motor controllers.

### Action → Joint Position Target

**File:** `g1_deploy_onnx_ref.cpp`, `CreatePolicyCommand()` (line 2805)

```cpp
for (int i = 0; i < G1_NUM_MOTOR; i++) {
    action_value = floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i];

    motor_command.q_target[i]  = default_angles[i] + action_value;  // position target (rad)
    motor_command.dq_target[i] = 0.0;                                // velocity target = 0
    motor_command.tau_ff[i]    = 0.0;                                // no feedforward torque
    motor_command.kp[i]        = kps[i];                             // stiffness gain
    motor_command.kd[i]        = kds[i];                             // damping gain
}
```

The conversion formula per joint:

```
q_target = default_angles[i] + action[i] × action_scale[i]
```

Where (from `policy_parameters.hpp`):
- **`default_angles`** (line 210) — standing pose joint offsets in radians (e.g., `-0.312`
  for hip pitch, `0.669` for knee)
- **`action_scale`** (line 109) — `0.25 × effort_limit / stiffness` per joint, mapping
  the network's [-1, 1]-ish output range to physically meaningful joint deltas
- **`isaaclab_to_mujoco`** (line 100) — reorders the 29 joints from IsaacLab training
  order (network output) to Unitree hardware order

### PD Control on the Robot

The motor command (`q_target`, `dq_target`, `kp`, `kd`, `tau_ff`) is published to the
Unitree G1's motor controllers at **500 Hz** via DDS (line 2493). Each motor runs a PD
control loop in firmware:

```
torque = kp × (q_target - q_actual) + kd × (dq_target - dq_actual) + tau_ff
```

Since `dq_target = 0` and `tau_ff = 0`, this simplifies to:

```
torque = kp × (q_target - q_actual) - kd × dq_actual
```

The `kp`/`kd` gains are computed from motor armature constants using a critically-damped
second-order model (`policy_parameters.hpp:17-22`):

```
stiffness (kp) = armature × ω²           where ω = 10 Hz × 2π
damping   (kd) = 2 × ζ × armature × ω   where ζ = 2.0
```

### Timing

| Loop | Rate | What it does |
|---|---|---|
| Encoder | 50 Hz | Gathers SMPL observations, runs encoder ONNX → 64-dim token |
| Policy | 50 Hz | Gathers robot state + token, runs decoder ONNX → 29 actions |
| Command writer | 500 Hz | Publishes latest motor command to robot via DDS |

The policy produces a new command at 50 Hz. The 500 Hz command writer re-publishes the
most recent command each tick, so the robot's PD controllers continuously track the
latest target between policy updates.
