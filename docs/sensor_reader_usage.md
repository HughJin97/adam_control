# 传感器读取模块使用说明

## 概述

传感器读取模块 (`sensor_reader.py`) 提供了从 MuJoCo 仿真环境读取机器人传感器数据并更新数据总线的功能。

## 主要功能

### 1. 读取的传感器数据类型

- **关节传感器**
  - 位置 (position) [rad]
  - 速度 (velocity) [rad/s]
  - 力矩 (torque) [Nm]
  - 关节限位信息

- **IMU传感器**
  - 姿态四元数 (orientation)
  - 欧拉角 (roll, pitch, yaw)
  - 角速度 (angular velocity) [rad/s]
  - 线加速度 (linear acceleration) [m/s²]

- **接触传感器**
  - 足部接触状态 (CONTACT/NO_CONTACT)
  - 接触力大小 [N]

- **计算的状态**
  - 质心位置和速度
  - 末端执行器（手脚）位置、速度、姿态

## 使用方法

### 基本使用

```python
import mujoco
from sensor_reader import create_sensor_reader, read_sensors
from data_bus import get_data_bus

# 1. 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)

# 2. 创建传感器读取器
sensor_reader = create_sensor_reader(model, data)

# 3. 在仿真循环中读取传感器
while simulation_running:
    # 读取所有传感器数据
    success = read_sensors(sensor_reader)
    
    if success:
        # 从数据总线获取数据
        data_bus = get_data_bus()
        
        # 获取关节位置
        hip_position = data_bus.get_joint_position("J_hip_l_roll")
        
        # 获取IMU数据
        imu_orientation = data_bus.get_imu_orientation()
        
        # 获取接触状态
        left_foot_contact = data_bus.get_end_effector_contact_state("left_foot")
    
    # 执行控制计算...
    # 执行仿真步...
    mujoco.mj_step(model, data)
```

### 高级功能

```python
# 获取详细的接触力信息
contact_forces = sensor_reader.get_contact_forces()
for foot_name, force in contact_forces.items():
    print(f"{foot_name} force: {force}")

# 检查关节限位状态
limits_status = sensor_reader.get_joint_limits_status()
for joint_name, status in limits_status.items():
    if status["near_lower_limit"]:
        print(f"{joint_name} is near lower limit!")
```

## 数据访问示例

### 关节数据
```python
# 获取单个关节数据
position = data_bus.get_joint_position("J_knee_l_pitch")
velocity = data_bus.get_joint_velocity("J_knee_l_pitch")
torque = data_bus.get_joint_torque("J_knee_l_pitch")

# 获取所有关节数据
all_positions = data_bus.get_all_joint_positions()
all_velocities = data_bus.get_all_joint_velocities()
```

### IMU数据
```python
# 获取姿态
orientation = data_bus.get_imu_orientation()  # 四元数
roll = data_bus.imu.roll  # 欧拉角
pitch = data_bus.imu.pitch
yaw = data_bus.imu.yaw

# 获取运动数据
angular_vel = data_bus.get_imu_angular_velocity()
linear_acc = data_bus.get_imu_linear_acceleration()
```

### 末端执行器数据
```python
# 获取足部位置
left_foot_pos = data_bus.get_end_effector_position("left_foot")
print(f"Left foot at: ({left_foot_pos.x}, {left_foot_pos.y}, {left_foot_pos.z})")

# 获取接触状态
contact_state = data_bus.get_end_effector_contact_state("left_foot")
if contact_state == ContactState.CONTACT:
    print("Left foot is in contact with ground")
```

### 质心数据
```python
# 获取质心状态
com_position = data_bus.get_center_of_mass_position()
com_velocity = data_bus.get_center_of_mass_velocity()
```

## 注意事项

1. **性能优化**：传感器ID在初始化时被缓存，避免重复查找
2. **线程安全**：数据总线使用递归锁保证线程安全
3. **错误处理**：如果传感器或关节不存在，会打印警告但不会中断执行
4. **数据更新**：每次调用 `read_sensors()` 会更新数据总线的时间戳

## 扩展功能

如需添加新的传感器类型，可以：

1. 在 `_init_sensor_mappings()` 中添加传感器名称
2. 创建新的读取方法（如 `_read_new_sensor()`）
3. 在 `read_sensors()` 中调用新方法
4. 在数据总线中添加相应的数据结构

## 完整示例

参见 `mujoco_integration_example.py` 获取完整的使用示例，包括：
- 仿真环境初始化
- 传感器数据读取
- 简单控制器实现
- 可视化和无头运行模式 