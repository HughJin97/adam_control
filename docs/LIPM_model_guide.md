# LIPM简化动力学模型使用指南

## 概述

本指南介绍如何使用基于Linear Inverted Pendulum Model (LIPM)的简化动力学模型。该模型采用单刚体近似，将机器人全身质量集中于质心，用于预测和控制机器人在支撑腿上的动态行为。

## 核心特性

- **单刚体建模**: 将复杂机器人简化为质心点模型
- **倒立摆动力学**: 实现经典LIPM方程 `ẍ = (g/z_c) * (x - p_foot)`
- **双控制模式**: 
  - 力控制模式：通过足底水平力控制质心
  - 步态控制模式：通过落脚位置规划控制质心
- **Pinocchio集成**: 自动从URDF模型提取机器人参数
- **数据总线集成**: 与现有系统无缝连接
- **轨迹预测**: 提供质心运动轨迹预测功能

## 数学模型

### 基本LIPM方程

质心在水平方向的动力学方程为：

```
ẍ = (g/z_c) * (x - p_foot_x)
ÿ = (g/z_c) * (y - p_foot_y)
```

其中：
- `(x, y)`: 质心水平位置
- `(p_foot_x, p_foot_y)`: 支撑足位置
- `z_c`: 质心高度
- `g`: 重力加速度

### 力控制扩展

在力控制模式下，方程扩展为：

```
ẍ = (g/z_c) * (x - p_foot_x) + f_x/m
ÿ = (g/z_c) * (y - p_foot_y) + f_y/m
```

其中：
- `(f_x, f_y)`: 足底水平控制力
- `m`: 机器人总质量

### 压力中心(COP)计算

从期望加速度反推所需COP位置：

```
COP_x = x - (z_c/g) * ẍ_desired
COP_y = y - (z_c/g) * ÿ_desired
```

## 快速开始

### 1. 基本使用

```python
from gait_core.simplified_dynamics import SimplifiedDynamicsModel
from gait_core.data_bus import Vector3D, get_data_bus

# 创建模型（使用默认参数）
lipm_model = SimplifiedDynamicsModel()

# 或从URDF文件加载参数
lipm_model = SimplifiedDynamicsModel(urdf_path="models/AzureLoong.urdf")

# 设置初始质心状态
data_bus = get_data_bus()
initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
initial_com_vel = Vector3D(x=0.1, y=0.0, z=0.0)

data_bus.set_center_of_mass_position(initial_com_pos)
data_bus.set_center_of_mass_velocity(initial_com_vel)

# 设置支撑足位置
support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
lipm_model.set_support_foot_position(support_foot)
```

### 2. 质心轨迹预测

```python
# 预测未来2秒的轨迹
time_horizon = 2.0
dt = 0.01
trajectory = lipm_model.predict_com_trajectory(time_horizon, dt)

# 提取轨迹数据
times = [i * dt for i in range(len(trajectory))]
x_positions = [state.com_position.x for state in trajectory]
y_positions = [state.com_position.y for state in trajectory]
```

### 3. 力控制模式

```python
# 设置控制力（向支撑足方向）
control_force = Vector3D(x=-20.0, y=-10.0, z=0.0)
lipm_model.set_foot_force_command(control_force)

# 预测受控轨迹
controlled_trajectory = lipm_model.predict_com_trajectory(2.0, 0.01)
```

### 4. 步态控制模式

```python
# 设置下一步落脚位置
next_footstep = Vector3D(x=0.2, y=0.1, z=0.0)
lipm_model.set_next_footstep(next_footstep)

# 实时更新模型
dt = 0.01
lipm_model.update(dt)
```

### 5. COP计算

```python
# 计算实现期望加速度的所需COP
desired_acceleration = Vector3D(x=-0.5, y=0.2, z=0.0)
required_cop = lipm_model.compute_required_cop(desired_acceleration)

print(f"所需COP位置: ({required_cop.x:.3f}, {required_cop.y:.3f})")
```

## 模型参数

### LIPMParameters类

```python
@dataclass
class LIPMParameters:
    total_mass: float = 50.0        # 机器人总质量 [kg]
    com_height: float = 0.8         # 质心高度 [m]
    gravity: float = 9.81           # 重力加速度 [m/s²]
    natural_frequency: float        # 自然频率 [rad/s] (自动计算)
    time_constant: float            # 时间常数 [s] (自动计算)
```

自然频率和时间常数自动计算：
- `natural_frequency = sqrt(gravity / com_height)`
- `time_constant = 1.0 / natural_frequency`

### 从URDF提取参数

模型可以自动从URDF文件提取以下参数：

```python
# 自动提取的参数
total_mass = pin.computeTotalMass(robot_model)  # 总质量
com_height = abs(com_position[2])              # 质心高度估计
```

## 状态和输入

### LIPMState类

```python
@dataclass
class LIPMState:
    com_position: Vector3D      # 质心位置 [m]
    com_velocity: Vector3D      # 质心速度 [m/s]
    com_acceleration: Vector3D  # 质心加速度 [m/s²]
    yaw: float                  # 偏航角 [rad]
    yaw_rate: float             # 偏航角速度 [rad/s]
    timestamp: float            # 时间戳
```

### LIPMInput类

```python
@dataclass
class LIPMInput:
    support_foot_position: Vector3D  # 支撑足位置 [m]
    foot_force: Vector3D            # 足底水平力 [N]
    next_footstep: Vector3D         # 下一步落脚位置 [m]
    control_mode: ControlMode       # 控制模式
```

## 控制模式

### 力控制模式 (FORCE_CONTROL)

通过设置足底水平力来控制质心运动：

```python
from gait_core.simplified_dynamics import ControlMode

# 设置力控制
control_force = Vector3D(x=-15.0, y=5.0, z=0.0)
lipm_model.set_foot_force_command(control_force)

# 检查控制模式
assert lipm_model.input_command.control_mode == ControlMode.FORCE_CONTROL
```

### 步态控制模式 (FOOTSTEP_CONTROL)

通过规划落脚位置来控制质心运动：

```python
# 设置步态控制
next_step = Vector3D(x=0.15, y=0.08, z=0.0)
lipm_model.set_next_footstep(next_step)

# 检查控制模式
assert lipm_model.input_command.control_mode == ControlMode.FOOTSTEP_CONTROL
```

## 与数据总线集成

模型与现有数据总线系统完全集成：

### 读取数据总线状态

```python
# 模型自动从数据总线读取：
# - 质心位置和速度
# - IMU姿态和角速度
# - 时间戳

lipm_model._update_state_from_data_bus()
```

### 写入预测结果

```python
# 模型自动将预测结果写入数据总线：
# - 预测的质心位置和速度
# - 可选的加速度预测

lipm_model._write_state_to_data_bus()
```

### 存储模型参数

```python
# 模型参数自动存储到数据总线
if hasattr(data_bus, 'lipm_parameters'):
    data_bus.lipm_parameters = {
        'total_mass': parameters.total_mass,
        'com_height': parameters.com_height,
        'natural_frequency': parameters.natural_frequency,
        # ...
    }
```

## 高级功能

### 行走步态模拟

```python
# 模拟行走步态
step_duration = 0.6    # 每步时间
step_length = 0.15     # 步长
step_width = 0.1       # 步宽

current_support_x = 0.0
for t in np.arange(0, time_horizon, dt):
    # 交替支撑腿
    if t % step_duration < step_duration / 2:
        current_support_y = step_width / 2   # 左脚支撑
    else:
        current_support_y = -step_width / 2  # 右脚支撑
    
    # 向前移动支撑足
    if t > 0 and t % step_duration < dt:
        current_support_x += step_length
    
    # 更新模型
    support_foot = Vector3D(x=current_support_x, y=current_support_y, z=0.0)
    lipm_model.set_support_foot_position(support_foot)
    lipm_model.update(dt)
```

### 稳定性分析

```python
# 检查平衡点稳定性
stable_pos = Vector3D(x=0.0, y=0.0, z=0.8)  # 支撑足正上方
stable_vel = Vector3D(x=0.0, y=0.0, z=0.0)

data_bus.set_center_of_mass_position(stable_pos)
data_bus.set_center_of_mass_velocity(stable_vel)

support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
lipm_model.set_support_foot_position(support_foot)

# 在平衡点，加速度应该为零
next_state = lipm_model.compute_com_dynamics(0.01)
print(f"平衡点加速度: ({next_state.com_acceleration.x:.6f}, {next_state.com_acceleration.y:.6f})")
```

## 运行示例

### 演示脚本

运行完整的演示程序：

```bash
cd /path/to/project
python examples/lipm_model_demo.py
```

演示包括：
1. 基本LIPM动力学演示
2. 力控制模式演示
3. 落脚点规划演示
4. URDF参数加载演示
5. COP计算演示

### 单元测试

运行测试验证实现正确性：

```bash
python tests/test_simplified_dynamics.py
```

测试覆盖：
- 参数初始化和计算
- 动力学方程正确性
- 数值积分精度
- 控制模式切换
- 数据总线集成
- 稳定性分析

## 性能考虑

### 计算复杂度

- **状态更新**: O(1) - 常数时间
- **轨迹预测**: O(n) - 线性于预测步数
- **URDF参数提取**: O(m) - 线性于关节数

### 数值稳定性

- 使用合理的时间步长 (dt ≤ 0.01s)
- 质心高度下限保护 (z_c ≥ 0.1m)
- 边界条件检查

### 实时性能

- 典型更新频率：100-1000 Hz
- 内存占用：< 1KB
- CPU使用：< 1%

## 限制和注意事项

### 模型限制

1. **单刚体假设**: 忽略了关节动力学和连杆惯性
2. **平面运动**: 假设质心高度基本恒定
3. **点接触**: 假设足部为点接触，忽略足底尺寸
4. **线性化**: 在小角度范围内有效

### 使用注意事项

1. **参数调优**: 根据实际机器人调整质量和高度参数
2. **时间步长**: 保持合理的时间步长以保证数值稳定性
3. **边界检查**: 监控质心位置和速度的合理范围
4. **传感器融合**: 结合IMU和视觉传感器提高状态估计精度

## 扩展和定制

### 添加新的控制模式

```python
class ControlMode(Enum):
    FORCE_CONTROL = 0
    FOOTSTEP_CONTROL = 1
    CUSTOM_CONTROL = 2  # 新增控制模式

# 在compute_com_dynamics中添加处理逻辑
if self.input_command.control_mode == ControlMode.CUSTOM_CONTROL:
    # 自定义控制逻辑
    pass
```

### 扩展状态向量

```python
@dataclass
class ExtendedLIPMState(LIPMState):
    angular_momentum: Vector3D = field(default_factory=Vector3D)
    energy: float = 0.0
    # 其他自定义状态变量
```

### 添加外部扰动

```python
def compute_com_dynamics_with_disturbance(self, dt: float, disturbance: Vector3D) -> LIPMState:
    # 在标准动力学基础上添加扰动
    next_state = self.compute_com_dynamics(dt)
    next_state.com_acceleration.x += disturbance.x / self.parameters.total_mass
    next_state.com_acceleration.y += disturbance.y / self.parameters.total_mass
    return next_state
```

## 参考资料

1. Kajita, S., et al. "Introduction to Humanoid Robotics" (2014)
2. Wieber, P. B. "Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations" (2006)
3. Koolen, T., et al. "Capturability-based Analysis and Control of Legged Locomotion" (2012)

## 支持和贡献

如有问题或建议，请联系开发团队或提交Issue。欢迎贡献代码和改进建议。 