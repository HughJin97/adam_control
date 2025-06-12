# 摆动足轨迹规划实现文档

## 概述

本项目实现了四足机器人摆动足轨迹规划的参数化描述方法，支持多种轨迹函数类型，能够为摆动足运动提供平滑、可控的运动轨迹。

## 功能特性

### 1. 支持的轨迹类型

#### 1.1 三次多项式插值 (Polynomial)
- **特点**: 平滑，计算简单，适合基础应用
- **数学基础**: 使用三次多项式进行水平插值，边界条件为起终点速度为0
- **公式**: `p(t) = at³ + bt² + ct + d`
- **优势**: 计算效率高，数学性质良好
- **适用场景**: 基础步态，稳定行走

#### 1.2 贝塞尔曲线 (Bezier)
- **特点**: 可控性好，形状优美，适合精确控制
- **数学基础**: 四点贝塞尔曲线，控制点可调节轨迹形状
- **公式**: `B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃`
- **优势**: 形状可控，视觉效果好
- **适用场景**: 精确控制，动态步态

#### 1.3 正弦函数 (Sinusoidal)
- **特点**: 自然，计算快速，适合实时应用
- **数学基础**: 正弦函数插值，可配置周期和平滑度
- **公式**: 水平方向 `sin(πt/2)`，垂直方向 `sin(πt)`
- **优势**: 自然流畅，计算快速
- **适用场景**: 实时控制，自然步态

#### 1.4 摆线轨迹 (Cycloid)
- **特点**: 独特形状，适合特殊需求
- **数学基础**: 摆线参数方程适配
- **公式**: `x = r(θ - sin(θ))`, `y = r(1 - cos(θ))`
- **优势**: 独特的运动特性
- **适用场景**: 特殊地形，爬坡步态

### 2. 轨迹参数化

#### 2.1 关键参数

| 参数名称 | 符号 | 建议范围 | 说明 |
|---------|------|----------|------|
| 抬脚高度 | h | 5-10 cm | 摆动足离地最大高度 |
| 摆动周期 | T | 0.2-0.6 s | 完整摆动相时间 |
| 最大高度相位 | φ_max | 0.45-0.5 | 达到最大高度的时间比例 |
| 地面间隙 | δ | 1-3 cm | 安全地面间隙 |

#### 2.2 轨迹特性

**水平方向 (X, Y)**:
- 平滑插值，确保起终点连续
- 支持不同插值函数（线性、多项式、正弦）
- 可添加小幅波动模拟自然性

**垂直方向 (Z)**:
- 拱形轨迹，在摆动中点达到最大高度
- 采用正弦函数: `z(t) = h × sin(πt)`
- 确保起终点接地 (z = 0)

### 3. 参数优化

#### 3.1 自适应抬脚高度
根据地形难度和行走速度自动调整：
```python
h_optimized = h_base + terrain_factor × 0.03 + speed_factor × 0.02
```

#### 3.2 时序优化
根据步态频率优化关键时间点：
- 摆动周期 = 摆动占比 / 步态频率
- 离地时间比例: 0.05 (快速离地)
- 最高点相位: 0.45 (稍早达到最高点)
- 触地时间比例: 0.95 (留出缓冲时间)

## 实现架构

### 1. 核心类

#### 1.1 SwingTrajectoryPlanner
主要轨迹规划器类，提供：
- 轨迹位置计算
- 轨迹速度计算
- 参数设置和管理
- 轨迹数据采样

#### 1.2 SwingTrajectoryConfig
配置参数类，包含：
- 轨迹类型选择
- 基本参数设置
- 速度约束
- 安全参数

#### 1.3 TrajectoryParameterOptimizer
参数优化器，提供：
- 抬脚高度优化
- 时序参数优化
- 地形适应调整

### 2. 文件结构

```
gait_core/
├── swing_trajectory.py          # 主要实现文件
config/
├── swing_trajectory_config.yaml # 配置文件
examples/
├── swing_trajectory_demo.py     # 演示脚本
tests/
├── test_swing_trajectory.py     # 单元测试
docs/
├── swing_trajectory_implementation.md # 本文档
```

## 使用示例

### 1. 基本使用

```python
from gait_core.swing_trajectory import create_swing_trajectory_planner
import numpy as np

# 创建轨迹规划器
planner = create_swing_trajectory_planner(
    trajectory_type="bezier",
    step_height=0.08,  # 8cm抬脚高度
    step_duration=0.4  # 0.4s摆动周期
)

# 设置起终点
start_pos = np.array([0.0, 0.1, 0.0])
end_pos = np.array([0.15, 0.1, 0.0])
planner.set_trajectory_parameters(start_pos, end_pos)

# 计算轨迹点
for phase in [0.0, 0.25, 0.5, 0.75, 1.0]:
    position = planner.compute_trajectory_position(phase)
    velocity = planner.compute_trajectory_velocity(phase)
    print(f"相位 {phase}: 位置{position}, 速度{velocity}")
```

### 2. 参数优化

```python
from gait_core.swing_trajectory import TrajectoryParameterOptimizer

optimizer = TrajectoryParameterOptimizer()

# 根据地形和速度优化抬脚高度
terrain_difficulty = 0.7  # 复杂地形
walking_speed = 1.0  # 1 m/s
optimized_height = optimizer.optimize_step_height(terrain_difficulty, walking_speed)

# 根据步态频率优化时序
gait_frequency = 1.5  # 1.5 Hz
timing_params = optimizer.optimize_trajectory_timing(gait_frequency)
```

### 3. 不同步态模式

```python
# 步行模式
walking_planner = create_swing_trajectory_planner("polynomial", 0.06, 0.5)

# 小跑模式
trotting_planner = create_swing_trajectory_planner("bezier", 0.08, 0.3)

# 跳跃模式
bounding_planner = create_swing_trajectory_planner("sinusoidal", 0.12, 0.25)

# 爬坡模式
climbing_planner = create_swing_trajectory_planner("cycloid", 0.10, 0.6)
```

## 配置参数

### 1. 基本参数配置

```yaml
# config/swing_trajectory_config.yaml
basic_parameters:
  step_height: 0.08               # 抬脚高度 [m]
  step_duration: 0.4              # 摆动周期 [s]
  ground_clearance: 0.02          # 地面间隙 [m]

timing_ratios:
  lift_off_ratio: 0.0             # 离地时刻
  max_height_ratio: 0.5           # 最大高度时刻
  touch_down_ratio: 1.0           # 着地时刻
```

### 2. 轨迹特定参数

```yaml
# 贝塞尔曲线参数
bezier_parameters:
  control_height_ratio: 0.8       # 控制点高度比例
  control_forward_ratio: 0.3      # 控制点前进比例

# 正弦函数参数
sinusoidal_parameters:
  vertical_periods: 1.0           # 垂直方向周期数
  horizontal_smooth: true         # 水平方向平滑
```

## 测试验证

### 1. 单元测试

运行完整测试套件：
```bash
python tests/test_swing_trajectory.py
```

测试覆盖：
- 轨迹类型正确性
- 边界条件验证
- 连续性检查
- 速度合理性
- 参数优化功能

### 2. 演示程序

运行可视化演示：
```bash
python examples/swing_trajectory_demo.py
```

演示内容：
- 四种轨迹类型对比
- 参数影响分析
- 优化效果展示

## 性能特性

### 1. 计算效率

| 轨迹类型 | 计算复杂度 | 实时性 | 内存占用 |
|---------|------------|--------|----------|
| Polynomial | O(1) | 优秀 | 最低 |
| Bezier | O(1) | 优秀 | 低 |
| Sinusoidal | O(1) | 优秀 | 低 |
| Cycloid | O(1) | 优秀 | 低 |

### 2. 轨迹质量

| 指标 | Polynomial | Bezier | Sinusoidal | Cycloid |
|------|------------|--------|------------|---------|
| 平滑度 | 优秀 | 优秀 | 良好 | 良好 |
| 可控性 | 良好 | 优秀 | 良好 | 中等 |
| 自然性 | 良好 | 优秀 | 优秀 | 中等 |
| 能耗效率 | 优秀 | 良好 | 优秀 | 中等 |

## 扩展指南

### 1. 添加新轨迹类型

1. 在 `TrajectoryType` 枚举中添加新类型
2. 在 `SwingTrajectoryPlanner` 中实现计算函数
3. 在配置文件中添加相关参数
4. 编写相应的单元测试

### 2. 集成到控制系统

```python
# 在步态控制器中集成
class GaitController:
    def __init__(self):
        self.swing_planner = create_swing_trajectory_planner("bezier")
    
    def update_swing_leg(self, swing_leg, phase):
        target_pos = self.swing_planner.compute_trajectory_position(phase)
        target_vel = self.swing_planner.compute_trajectory_velocity(phase)
        # 发送到关节控制器
        self.send_leg_commands(swing_leg, target_pos, target_vel)
```

## 总结

本实现提供了完整的摆动足轨迹规划解决方案，具有以下特点：

1. **多样性**: 支持四种不同的轨迹函数类型
2. **参数化**: 关键参数可配置和优化
3. **实用性**: 适应不同步态模式和地形条件
4. **可扩展**: 模块化设计，易于扩展新功能
5. **高质量**: 完整的测试验证和文档支持

关键参数建议：
- **抬脚高度 h**: 5-10cm，根据地形调整
- **最大高度相位**: 0.5（摆动中点）或 0.45（稍早）
- **轨迹函数选择**: 
  - 基础应用 → Polynomial
  - 精确控制 → Bezier  
  - 实时应用 → Sinusoidal
  - 特殊需求 → Cycloid

该实现为四足机器人提供了可靠的摆动足运动轨迹，能够满足不同应用场景的需求。 