# 控制发送模块使用说明

## 概述

控制发送模块 (`control_sender.py`) 负责从数据总线读取控制指令并发送到 MuJoCo 仿真环境，支持多种控制模式。

## 主要功能

### 1. 支持的控制模式

- **力矩控制 (TORQUE)**
  - 直接发送关节力矩指令
  - 适合精确的力控制场景
  - 从 `DataBus.control_torques` 读取

- **位置控制 (POSITION)**
  - 发送期望关节位置
  - 内部使用PD控制器计算力矩
  - 从 `DataBus.control_positions` 读取

- **速度控制 (VELOCITY)**
  - 发送期望关节速度
  - 内部使用P控制器计算力矩
  - 从 `DataBus.control_velocities` 读取

- **混合控制 (HYBRID)**
  - 每个关节可以独立设置控制模式
  - 灵活组合不同的控制策略

### 2. 安全特性

- 自动从模型读取并应用关节限制
- 力矩、位置、速度限制检查
- 可开关的安全检查功能

## 使用方法

### 基本使用

```python
import mujoco
from control_sender import create_control_sender, send_commands, ControlMode
from data_bus import get_data_bus

# 1. 创建控制发送器
model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)
control_sender = create_control_sender(model, data)

# 2. 设置控制模式
control_sender.set_control_mode(ControlMode.TORQUE)

# 3. 在控制循环中
data_bus = get_data_bus()
while simulation_running:
    # 读取传感器...
    
    # 计算控制输出并写入数据总线
    for joint_name in joint_names:
        torque = compute_torque(joint_name)  # 你的控制算法
        data_bus.set_control_torque(joint_name, torque)
    
    # 发送控制指令
    success = send_commands(control_sender)
    
    # 执行仿真步
    mujoco.mj_step(model, data)
```

### 位置控制模式

```python
# 设置位置控制模式
control_sender.set_control_mode(ControlMode.POSITION)

# 更新PD增益（可选）
control_sender.update_control_gains("J_knee_l_pitch", kp=200.0, kd=20.0)

# 设置期望位置
data_bus.set_control_position("J_knee_l_pitch", 0.5)  # 弧度

# 发送指令（内部会计算PD控制）
send_commands(control_sender)
```

### 混合控制模式

```python
# 设置混合控制模式
control_sender.set_control_mode(ControlMode.HYBRID)

# 为不同关节设置不同的控制模式
control_sender.set_joint_control_mode("J_hip_l_pitch", ControlMode.POSITION)
control_sender.set_joint_control_mode("J_knee_l_pitch", ControlMode.POSITION)
control_sender.set_joint_control_mode("J_arm_l_01", ControlMode.TORQUE)

# 分别设置控制指令
data_bus.set_control_position("J_hip_l_pitch", -0.2)
data_bus.set_control_position("J_knee_l_pitch", 0.4)
data_bus.set_control_torque("J_arm_l_01", 5.0)

# 发送混合指令
send_commands(control_sender)
```

## 控制流程

### 力矩控制流程
```
DataBus.control_torques → 安全检查 → MuJoCo actuator.ctrl
```

### 位置控制流程
```
DataBus.control_positions → PD控制器 → 力矩计算 → 安全检查 → MuJoCo actuator.ctrl
```

### 速度控制流程
```
DataBus.control_velocities → P控制器 → 力矩计算 → 安全检查 → MuJoCo actuator.ctrl
```

## 高级功能

### 1. 安全检查控制

```python
# 禁用安全检查（谨慎使用）
control_sender.set_safety_checks(False)

# 重新启用
control_sender.set_safety_checks(True)
```

### 2. 获取执行器信息

```python
# 获取所有执行器的当前状态
actuator_info = control_sender.get_actuator_info()
for joint_name, info in actuator_info.items():
    print(f"{joint_name}:")
    print(f"  Actuator ID: {info['actuator_id']}")
    print(f"  Current control: {info['current_control']}")
    print(f"  Control mode: {info['control_mode']}")
```

### 3. 重置控制指令

```python
# 将所有控制指令重置为零
control_sender.reset_commands()
```

### 4. 自定义控制增益

```python
# 为特定关节设置控制增益
control_sender.update_control_gains(
    joint_name="J_hip_l_pitch",
    kp=150.0,  # 位置增益
    kd=15.0    # 速度增益
)
```

## 默认控制增益

模块根据关节类型预设了不同的控制增益：

- **髋关节 (hip)**: kp=150, kd=15
- **膝关节 (knee)**: kp=120, kd=12
- **踝关节 (ankle)**: kp=80, kd=8
- **手臂关节 (arm)**: kp=50, kd=5
- **其他关节**: kp=100, kd=10

## 注意事项

1. **执行器映射**：模块自动建立关节到执行器的映射关系
2. **控制频率**：建议以1000Hz运行控制循环
3. **单位**：所有角度使用弧度制，力矩单位为Nm
4. **安全限制**：默认启用，从MuJoCo模型自动读取

## 典型应用场景

### 1. 平衡站立（力矩控制）
```python
# 使用IMU反馈进行平衡控制
imu_roll = data_bus.imu.roll
balance_torque = -kp * imu_roll - kd * imu_angular_velocity
data_bus.set_control_torque("J_hip_l_roll", balance_torque)
```

### 2. 轨迹跟踪（位置控制）
```python
# 跟踪预定义的关节轨迹
time_now = data.time
desired_pos = trajectory_function(time_now)
data_bus.set_control_position("J_knee_l_pitch", desired_pos)
```

### 3. 步态控制（混合模式）
```python
# 支撑腿用力矩控制，摆动腿用位置控制
if leg_in_stance:
    control_sender.set_joint_control_mode(joint, ControlMode.TORQUE)
    data_bus.set_control_torque(joint, stance_torque)
else:
    control_sender.set_joint_control_mode(joint, ControlMode.POSITION)
    data_bus.set_control_position(joint, swing_position)
```

## 完整示例

参见 `mujoco_integration_example.py` 获取完整的使用示例，包括：
- 不同控制模式的切换
- 与传感器读取模块的配合
- 实时控制循环的实现
- 控制模式演示功能 