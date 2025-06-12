# 单脚平衡仿真系统

## 概述

本项目实现了在Mujoco环境中的机器人单脚支撑平衡仿真，集成了MPC（模型预测控制）算法来计算和维持平衡。机器人能够保持单脚支撑静止状态，通过实时控制算法调整姿态以维持稳定性。

## 🎯 主要功能

- **单脚支撑平衡**: 机器人可以使用左脚或右脚单独支撑
- **实时平衡控制**: 125Hz控制频率，1ms物理仿真步长
- **3D可视化**: Mujoco viewer实时显示机器人状态
- **数据记录与分析**: 记录质心位置、平衡误差、控制力等关键数据
- **结果可视化**: 自动生成仿真结果图表

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.8+
pip install mujoco scipy matplotlib numpy
```

### 2. 运行演示

```bash
# 交互式演示（推荐）
python demo_single_foot_balance.py

# 简化版启动脚本
python run_simple_balance.py

# 直接命令行运行
python sim/simple_single_foot_balance.py --support-foot right --duration 10 --plot
```

### 3. 基本用法

```python
from sim.simple_single_foot_balance import SimpleSingleFootBalanceSimulation

# 创建仿真实例
sim = SimpleSingleFootBalanceSimulation()
sim.support_foot = 'right'  # 或 'left'

# 运行仿真
sim.run_simulation(duration=10.0, use_viewer=True)

# 查看结果
sim.plot_results()
```

## 📁 文件结构

```
├── sim/
│   ├── simple_single_foot_balance.py    # 简化版单脚平衡仿真
│   └── single_foot_balance_simulation.py # 完整版MPC集成仿真
├── models/
│   ├── scene.xml                        # 主场景文件
│   └── AzureLoong.xml                   # 机器人模型
├── docs/
│   └── Single_Foot_Balance_Simulation_Guide.md # 详细使用指南
├── demo_single_foot_balance.py          # 演示脚本
├── run_simple_balance.py               # 简化启动脚本
└── README_Single_Foot_Balance.md       # 本文件
```

## 🔧 技术实现

### 控制算法

1. **PD控制器**: 维持关节目标角度
2. **平衡控制**: 踝关节补偿质心偏移
3. **力控制**: 计算期望地面反力
4. **状态估计**: 实时获取机器人状态

### 关键参数

```python
# 仿真参数
dt = 0.001              # 1ms仿真步长
control_dt = 0.008      # 8ms控制步长 (125Hz)

# 控制器参数
kp_joint = 100.0        # 关节位置增益
kd_joint = 10.0         # 关节速度增益
kp_balance = 500.0      # 平衡控制增益
kd_balance = 50.0       # 平衡阻尼增益

# 机器人参数
robot_mass = 45.0       # kg
gravity = 9.81          # m/s²
```

### 单脚支撑姿态

#### 右脚支撑
- **支撑腿（右腿）**: 轻微弯曲，提供稳定性
- **摆动腿（左腿）**: 大腿抬起，小腿弯曲
- **手臂**: 左臂向前，右臂向后，辅助平衡

#### 左脚支撑
- **支撑腿（左腿）**: 轻微弯曲，提供稳定性
- **摆动腿（右腿）**: 大腿抬起，小腿弯曲
- **手臂**: 右臂向前，左臂向后，辅助平衡

## 📊 仿真结果

### 性能指标
- **仿真频率**: 1000Hz (1ms步长)
- **控制频率**: 125Hz (8ms步长)
- **平衡精度**: 通常 < 1m (简化算法)
- **稳定时间**: < 2秒

### 输出数据
1. **质心位置**: X、Y、Z方向的轨迹
2. **平衡误差**: 相对支撑脚的偏移
3. **控制力**: 期望地面反力
4. **关节力矩**: 各关节的控制力矩
5. **质心轨迹**: XY平面的运动轨迹

## 🎮 使用示例

### 演示模式

```bash
python demo_single_foot_balance.py
```

选择演示模式：
1. **完整演示**: 包含左脚和右脚支撑
2. **快速演示**: 仅右脚支撑，5秒演示

### 命令行模式

```bash
# 右脚支撑，10秒仿真，显示可视化和结果图表
python sim/simple_single_foot_balance.py --support-foot right --duration 10 --plot

# 左脚支撑，5秒仿真，无可视化界面
python sim/simple_single_foot_balance.py --support-foot left --duration 5 --no-viewer

# 查看所有选项
python sim/simple_single_foot_balance.py --help
```

### 交互式模式

```bash
python run_simple_balance.py
```

按提示选择：
- 支撑脚（左脚/右脚）
- 仿真时长
- 是否显示3D可视化
- 是否绘制结果图表

## 🔍 MPC集成版本

项目还包含完整的MPC集成版本（需要额外依赖）：

```python
from sim.single_foot_balance_simulation import SingleFootBalanceSimulation
# 完整的MPC数据总线集成
# 支持多种输出模式和高级控制功能
```

详见：[MPC数据总线集成指南](docs/MPC_DataBus_Integration_Guide.md)

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   ```
   检查 models/scene.xml 和 models/AzureLoong.xml 是否存在
   ```

2. **机器人倒下**
   ```
   调整控制参数 kp_balance 和 kd_balance
   ```

3. **仿真运行缓慢**
   ```
   关闭可视化界面：--no-viewer
   ```

4. **依赖包缺失**
   ```bash
   pip install mujoco scipy matplotlib numpy
   ```

### 调试技巧

```python
# 检查机器人状态
robot_state = sim.get_robot_state()
print(f"质心位置: {robot_state['com_position']}")
print(f"支撑脚位置: {robot_state['support_foot_position']}")

# 监控平衡误差
balance_control = sim.compute_balance_control(robot_state)
print(f"平衡误差: {balance_control['balance_error']}")
```

## 📈 扩展开发

### 添加新的平衡策略

```python
def custom_balance_control(self, robot_state):
    # 实现自定义平衡算法
    com_pos = robot_state['com_position']
    foot_pos = robot_state['support_foot_position']
    
    # 计算平衡误差
    error = com_pos[:2] - foot_pos[:2]
    
    # 自定义控制逻辑
    control_force = -self.kp * error
    
    return {
        'balance_error': error,
        'desired_force': control_force
    }
```

### 修改机器人姿态

```python
def custom_stance_pose(self):
    pose = {}
    # 定义自定义关节角度
    pose['J_hip_r_pitch'] = -0.2  # 更大的前倾
    pose['J_knee_r_pitch'] = 0.4  # 更大的弯曲
    # ... 其他关节
    return pose
```

### 集成其他控制器

```python
from your_controller import AdvancedBalanceController

sim = SimpleSingleFootBalanceSimulation()
sim.balance_controller = AdvancedBalanceController()
```

## 📚 相关文档

- [单脚平衡仿真指南](docs/Single_Foot_Balance_Simulation_Guide.md)
- [MPC数据总线集成指南](docs/MPC_DataBus_Integration_Guide.md)
- [Mujoco仿真环境配置](docs/Mujoco_Setup_Guide.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证。

---

**开发者**: Adam Control Team  
**最后更新**: 2024年  
**版本**: 1.0.0 