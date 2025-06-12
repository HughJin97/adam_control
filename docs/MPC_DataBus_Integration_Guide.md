# MPC数据总线集成指南

## 概述

本指南详细介绍了如何将MPC（模型预测控制）求解结果集成到机器人数据总线系统中，实现MPC输出与其他控制模块的无缝对接。

## 功能特性

### 1. 多种输出模式
- **FORCE_ONLY**: 仅输出支撑力到 `DataBus.desired_force[foot]`
- **TRAJECTORY_ONLY**: 仅输出质心轨迹供WBC跟踪
- **FOOTSTEP_ONLY**: 仅输出智能落脚点到 `DataBus.target_foot_pos`
- **COMBINED**: 组合输出（推荐）

### 2. 数据输出类型

#### 支撑力输出
- **目标**: `DataBus.desired_force[foot_name]`
- **格式**: `Vector3D(fx, fy, fz)` [N]
- **特性**: 
  - 力平滑处理（可配置平滑系数）
  - 力阈值过滤（小于阈值的力设为0）
  - 实时更新（可配置更新频率）

#### 质心轨迹输出
- **位置**: `DataBus.center_of_mass.position`
- **速度**: `DataBus.center_of_mass.velocity`
- **加速度**: `DataBus.center_of_mass.acceleration`
- **轨迹缓存**: `DataBus.mpc_com_trajectory`
- **ZMP轨迹**: `DataBus.mpc_zmp_trajectory`

#### 落脚点输出
- **目标**: `DataBus.target_foot_pos[foot_name]`
- **格式**: `{'x': float, 'y': float, 'z': float}` [m]
- **特性**:
  - 智能更新阈值（避免频繁小幅调整）
  - 安全裕量检查
  - 与现有步态规划器兼容

### 3. MPC状态监控
- **求解状态**: 成功/失败标志
- **求解时间**: 实时性能监控
- **优化代价**: 控制质量评估
- **求解器类型**: OSQP/qpOASES/CVXPY/SciPy

## 核心组件

### 1. MPCDataBusIntegrator
主要集成器类，负责将MPC结果写入数据总线。

```python
from gait_core.mpc_data_bus_integration import (
    MPCDataBusIntegrator, MPCOutputMode, MPCDataBusConfig
)

# 创建集成器
config = MPCDataBusConfig(
    output_mode=MPCOutputMode.COMBINED,
    update_frequency=125.0,  # Hz
    force_smoothing_factor=0.8,
    enable_data_validation=True
)

integrator = MPCDataBusIntegrator(data_bus, config)
```

### 2. 数据总线MPC接口
扩展的数据总线接口，专门支持MPC数据。

```python
# 期望接触力
data_bus.set_desired_contact_force('left_foot', Vector3D(10, 5, 80))
force = data_bus.get_desired_contact_force('left_foot')

# 质心加速度
data_bus.set_center_of_mass_acceleration(Vector3D(0.1, 0.05, 0))
acc = data_bus.get_center_of_mass_acceleration()

# MPC状态
data_bus.update_mpc_status(solve_time=0.015, success=True, cost=125.6, solver_type='osqp')
status = data_bus.get_mpc_status()

# 轨迹缓存
data_bus.set_mpc_com_trajectory([Vector3D(0.1, 0, 0.8), Vector3D(0.15, 0.02, 0.8)])
trajectory = data_bus.get_mpc_com_trajectory()
```

## 使用示例

### 基本集成流程

```python
# 1. 创建数据总线和集成器
data_bus = get_data_bus()
integrator = create_mpc_databus_integrator(
    data_bus,
    output_mode=MPCOutputMode.COMBINED,
    update_frequency=125.0
)

# 2. MPC求解循环
for step in range(simulation_steps):
    current_time = step * dt
    
    # MPC求解
    mpc_result = mpc_solver.solve(current_state, gait_plan)
    
    # 集成到数据总线
    success = integrator.update_from_mpc_result(mpc_result, current_time)
    
    if success:
        # 其他模块可以从数据总线读取MPC输出
        desired_forces = data_bus.get_all_desired_contact_forces()
        com_position = data_bus.get_center_of_mass_position()
        target_positions = data_bus.get_target_foot_position('left_foot')
```

### 不同输出模式示例

```python
# 仅力输出模式（用于力控制）
force_integrator = create_mpc_databus_integrator(
    data_bus, 
    output_mode=MPCOutputMode.FORCE_ONLY
)

# 仅轨迹输出模式（用于轨迹跟踪）
trajectory_integrator = create_mpc_databus_integrator(
    data_bus,
    output_mode=MPCOutputMode.TRAJECTORY_ONLY
)

# 仅落脚点输出模式（用于步态规划）
footstep_integrator = create_mpc_databus_integrator(
    data_bus,
    output_mode=MPCOutputMode.FOOTSTEP_ONLY
)
```

## 配置参数

### MPCDataBusConfig 参数说明

```python
@dataclass
class MPCDataBusConfig:
    # 输出模式
    output_mode: MPCOutputMode = MPCOutputMode.COMBINED
    
    # 力输出配置
    force_output_enabled: bool = True
    force_smoothing_factor: float = 0.8    # 力平滑系数 (0-1)
    force_threshold: float = 1.0           # 最小输出力阈值 [N]
    
    # 轨迹输出配置
    trajectory_output_enabled: bool = True
    trajectory_lookahead_steps: int = 3    # 轨迹前瞻步数
    com_velocity_scaling: float = 1.0      # 质心速度缩放
    
    # 落脚点输出配置
    footstep_output_enabled: bool = True
    footstep_update_threshold: float = 0.05 # 落脚点更新阈值 [m]
    footstep_safety_margin: float = 0.02   # 安全裕量 [m]
    
    # 更新频率控制
    update_frequency: float = 125.0        # 数据更新频率 [Hz]
    force_update_rate: float = 250.0       # 力更新频率 [Hz]
    trajectory_update_rate: float = 100.0  # 轨迹更新频率 [Hz]
    
    # 数据验证
    enable_data_validation: bool = True
    max_force_magnitude: float = 300.0     # 最大力幅值 [N]
    max_com_velocity: float = 2.0          # 最大质心速度 [m/s]
```

## 性能特性

### 实时性能
- **更新频率**: 125-250 Hz
- **集成延迟**: < 1ms
- **内存占用**: 最小化轨迹缓存

### 数据验证
- **力限制检查**: 防止异常大力输出
- **速度限制检查**: 确保合理的质心运动
- **求解状态验证**: 仅处理成功的MPC结果

### 平滑处理
- **指数平滑**: 减少力输出的高频噪声
- **阈值过滤**: 消除微小的无效力
- **频率控制**: 避免过度频繁的更新

## 与其他模块的集成

### 1. 与WBC的集成
```python
# WBC可以直接从数据总线读取质心轨迹
com_position_ref = data_bus.get_center_of_mass_position()
com_velocity_ref = data_bus.get_center_of_mass_velocity()
com_acceleration_ref = data_bus.get_center_of_mass_acceleration()

# 用于WBC的轨迹跟踪
wbc_controller.set_com_reference(com_position_ref, com_velocity_ref, com_acceleration_ref)
```

### 2. 与力控制器的集成
```python
# 力控制器读取期望接触力
left_force_ref = data_bus.get_desired_contact_force('left_foot')
right_force_ref = data_bus.get_desired_contact_force('right_foot')

# 应用到关节力矩控制
force_controller.set_contact_force_reference(left_force_ref, right_force_ref)
```

### 3. 与步态规划器的集成
```python
# 步态规划器可以使用MPC的智能落脚点
left_target = data_bus.get_target_foot_position('left_foot')
right_target = data_bus.get_target_foot_position('right_foot')

# 更新步态规划目标
gait_planner.update_foot_targets(left_target, right_target)
```

## 监控和调试

### 1. 集成统计
```python
# 获取集成性能统计
stats = integrator.get_integration_statistics()
print(f"更新次数: {stats['update_count']}")
print(f"平均更新时间: {stats['average_update_time']:.3f}ms")
print(f"成功率: {stats['success_rate']:.1%}")
print(f"验证错误: {stats['validation_errors']}")
```

### 2. MPC状态监控
```python
# 监控MPC求解状态
mpc_status = data_bus.get_mpc_status()
print(f"求解成功: {mpc_status['solve_success']}")
print(f"求解时间: {mpc_status['last_solve_time']:.3f}s")
print(f"优化代价: {mpc_status['cost']:.2f}")
print(f"求解器: {mpc_status['solver_type']}")
```

### 3. 数据摘要
```python
# 获取MPC数据摘要
summary = data_bus.get_mpc_data_summary()
print(f"期望力数量: {summary['desired_forces_count']}")
print(f"质心轨迹长度: {summary['com_trajectory_length']}")
print(f"ZMP轨迹长度: {summary['zmp_trajectory_length']}")
print(f"目标足部位置: {len(summary['target_foot_positions'])}")
```

## 故障排除

### 常见问题

1. **集成失败**
   - 检查MPC求解是否成功
   - 验证数据总线连接
   - 确认配置参数正确

2. **更新频率过低**
   - 调整 `update_frequency` 参数
   - 检查MPC求解时间
   - 优化数据验证设置

3. **力输出异常**
   - 检查 `max_force_magnitude` 限制
   - 调整 `force_smoothing_factor`
   - 验证 `force_threshold` 设置

4. **轨迹不连续**
   - 检查MPC预测时域设置
   - 验证质心轨迹数据完整性
   - 调整轨迹更新频率

### 调试技巧

1. **启用详细日志**
```python
config.enable_data_validation = True
# 查看详细的验证错误信息
```

2. **降低更新频率进行调试**
```python
config.update_frequency = 10.0  # 降低到10Hz进行调试
```

3. **使用单一输出模式测试**
```python
# 逐个测试不同输出模式
config.output_mode = MPCOutputMode.FORCE_ONLY
```

## 测试验证

### 单元测试
运行基础集成测试：
```bash
PYTHONPATH=/path/to/adam_control python tests/test_mpc_databus_integration_basic.py
```

### 集成测试
运行完整集成演示：
```bash
PYTHONPATH=/path/to/adam_control python examples/mpc_databus_integration_demo.py
```

### 测试覆盖
- ✅ 基本集成功能
- ✅ 不同输出模式
- ✅ 数据验证
- ✅ 频率控制
- ✅ 错误处理
- ✅ 统计监控
- ✅ 数据总线接口

## 最佳实践

### 1. 配置建议
- 使用 `COMBINED` 模式获得最佳功能
- 设置合理的更新频率（100-250 Hz）
- 启用数据验证确保安全性
- 调整平滑参数减少噪声

### 2. 性能优化
- 避免过高的更新频率
- 合理设置轨迹缓存长度
- 使用高效的MPC求解器（OSQP推荐）
- 定期清理历史数据

### 3. 安全考虑
- 设置合理的力和速度限制
- 启用数据验证
- 监控集成成功率
- 实现故障恢复机制

## 总结

MPC数据总线集成系统提供了完整的解决方案，将MPC求解结果无缝集成到机器人控制系统中。通过灵活的配置选项、强大的数据验证和实时性能监控，确保了系统的可靠性和高效性。

主要优势：
- **模块化设计**: 支持不同输出模式和灵活配置
- **实时性能**: 高频率更新和低延迟集成
- **数据安全**: 完整的验证和限制检查
- **易于集成**: 与现有控制模块无缝对接
- **监控完善**: 详细的统计和调试信息

通过本指南的实施，可以实现高质量的MPC控制系统集成，为机器人的稳定行走和动态控制提供强有力的支持。 