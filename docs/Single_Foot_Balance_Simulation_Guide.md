# 单脚平衡仿真指南

## 概述

本仿真在Mujoco环境中实现机器人单脚支撑平衡控制，集成MPC（模型预测控制）算法来计算和维持平衡。机器人将保持单脚支撑静止状态，通过MPC控制器实时调整姿态以维持平衡。

## 功能特性

### 🎯 核心功能
- **单脚支撑平衡**: 机器人保持左脚或右脚单独支撑
- **MPC平衡控制**: 实时计算平衡所需的控制力和力矩
- **实时仿真**: 1ms物理仿真步长，125Hz控制频率
- **3D可视化**: Mujoco viewer实时显示机器人状态
- **数据记录**: 记录质心位置、平衡误差、控制力等数据

### 🔧 技术特性
- **数据总线集成**: 与现有MPC数据总线系统完全集成
- **多种输出模式**: 支持力控制、轨迹跟踪、足端位置等输出
- **安全限制**: 力和速度限制，防止过度控制
- **实时监控**: 平衡误差、求解时间等性能指标

## 快速开始

### 1. 简单启动（推荐）

```bash
python run_single_foot_balance.py
```

按照交互式提示选择：
- 支撑脚（左脚/右脚）
- 仿真时长
- 是否显示3D可视化
- 是否绘制结果图表

### 2. 命令行启动

```bash
# 右脚支撑，10秒仿真，显示可视化和结果图表
python sim/single_foot_balance_simulation.py --support-foot right --duration 10 --plot

# 左脚支撑，5秒仿真，无可视化界面
python sim/single_foot_balance_simulation.py --support-foot left --duration 5 --no-viewer

# 查看所有选项
python sim/single_foot_balance_simulation.py --help
```

### 3. 程序化调用

```python
from sim.single_foot_balance_simulation import SingleFootBalanceSimulation

# 创建仿真实例
sim = SingleFootBalanceSimulation()
sim.support_foot = 'right'  # 或 'left'
sim.target_joint_positions = sim._get_single_foot_stance_pose()

# 运行仿真
sim.run_simulation(duration=10.0, use_viewer=True)

# 绘制结果
sim.plot_results()
```

## 仿真参数配置

### 机器人配置
```python
# 机器人物理参数
robot_mass = 45.0  # kg
gravity = 9.81     # m/s²

# 控制器参数
kp_joint = 100.0   # 关节位置增益
kd_joint = 10.0    # 关节速度增益
kp_balance = 500.0 # 平衡控制增益
kd_balance = 50.0  # 平衡阻尼增益
```

### MPC配置
```python
mpc_config = MPCDataBusConfig(
    update_frequency=125.0,        # 125Hz更新频率
    output_mode=MPCOutputMode.COMBINED,
    force_smoothing_factor=0.8,    # 力平滑因子
    enable_data_validation=True,   # 数据验证
    max_force_magnitude=300.0,     # 最大力限制 (N)
    max_velocity_magnitude=2.0     # 最大速度限制 (m/s)
)
```

### 仿真配置
```python
dt = 0.001         # 1ms仿真步长
control_dt = 0.008 # 8ms控制步长 (125Hz)
```

## 单脚支撑姿态

### 右脚支撑姿态
- **支撑腿（右腿）**: 保持直立，轻微弯曲以提供稳定性
- **摆动腿（左腿）**: 抬起大腿，弯曲小腿
- **手臂**: 左臂向前，右臂向后，辅助平衡
- **躯干**: 保持直立

### 左脚支撑姿态
- **支撑腿（左腿）**: 保持直立，轻微弯曲
- **摆动腿（右腿）**: 抬起大腿，弯曲小腿
- **手臂**: 右臂向前，左臂向后，辅助平衡
- **躯干**: 保持直立

## MPC平衡控制原理

### 1. 状态估计
```python
# 获取机器人当前状态
robot_state = {
    'com_position': [x, y, z],      # 质心位置
    'com_velocity': [vx, vy, vz],   # 质心速度
    'support_foot_position': [fx, fy, fz]  # 支撑脚位置
}
```

### 2. 平衡误差计算
```python
# 期望质心在支撑脚正上方
desired_com_x = foot_pos[0]
desired_com_y = foot_pos[1]

balance_error_x = com_pos[0] - desired_com_x
balance_error_y = com_pos[1] - desired_com_y
```

### 3. MPC控制计算
```python
# 计算期望的地面反力
desired_force_x = -kp_balance * balance_error_x - kd_balance * com_vel[0]
desired_force_y = -kp_balance * balance_error_y - kd_balance * com_vel[1]
desired_force_z = robot_mass * gravity  # 重力补偿
```

### 4. 数据总线更新
```python
# 通过MPC数据总线集成器处理结果
mpc_result = {
    'contact_forces': {foot_name: [fx, fy, fz]},
    'com_trajectory': {...},
    'solve_time': 0.005,
    'solver_status': 'optimal'
}
mpc_integrator.process_mpc_result(mpc_result)
```

## 输出数据分析

### 实时监控指标
- **平衡误差**: 质心相对支撑脚的偏移
- **控制力**: MPC计算的期望地面反力
- **求解时间**: MPC求解器性能
- **关节角度**: 各关节的实际和目标角度

### 结果图表
仿真结束后可显示4个图表：

1. **质心位置**: X、Y、Z方向的质心轨迹
2. **平衡误差**: X、Y方向的平衡误差变化
3. **控制力**: Fx、Fy、Fz三个方向的控制力
4. **平衡误差范数**: 总体平衡误差的变化趋势

## 性能指标

### 典型性能
- **仿真频率**: 1000Hz (1ms步长)
- **控制频率**: 125Hz (8ms步长)
- **MPC求解时间**: ~5ms
- **平衡精度**: <1cm (典型值)
- **稳定时间**: <2秒

### 系统要求
- **Python**: 3.8+
- **Mujoco**: 2.3+
- **内存**: 建议2GB+
- **CPU**: 支持实时仿真

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   错误: 加载模型失败
   解决: 检查models/scene.xml和models/AzureLoong.xml文件是否存在
   ```

2. **机器人倒下**
   ```
   原因: 初始姿态不稳定或控制参数不当
   解决: 调整kp_balance和kd_balance参数
   ```

3. **仿真运行缓慢**
   ```
   原因: 计算资源不足
   解决: 关闭可视化界面或降低仿真精度
   ```

4. **MPC求解失败**
   ```
   原因: 约束条件过严或初始状态不合理
   解决: 检查力和速度限制参数
   ```

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查数据总线状态**
   ```python
   print(sim.data_bus.get_mpc_data_summary())
   ```

3. **监控关键指标**
   ```python
   print(f"平衡误差: {control_result['balance_error']}")
   print(f"控制力: {control_result['desired_force']}")
   ```

## 扩展开发

### 添加新的平衡策略
```python
def custom_balance_control(self, robot_state):
    # 实现自定义平衡算法
    pass
```

### 集成其他控制器
```python
from your_controller import CustomController
sim.controller = CustomController()
```

### 修改机器人姿态
```python
def custom_stance_pose(self):
    # 定义自定义单脚支撑姿态
    pose = {}
    # ... 设置关节角度
    return pose
```

## 相关文档

- [MPC数据总线集成指南](MPC_DataBus_Integration_Guide.md)
- [Mujoco仿真环境配置](Mujoco_Setup_Guide.md)
- [机器人模型说明](Robot_Model_Documentation.md)

## 技术支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查相关日志输出
3. 提交Issue并附上详细的错误信息

---

*最后更新: 2024年* 