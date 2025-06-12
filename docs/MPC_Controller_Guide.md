# MPC控制器使用指南

## 概述

基于线性倒立摆模型（LIPM）的模型预测控制器（MPC Controller）实现了机器人质心轨迹跟踪和步态规划的优化控制。该控制器通过求解有约束的优化问题，同时优化足底力和落脚点位置，实现稳定的机器人行走控制。

## 核心特性

### 1. 优化问题定义

#### 时域设置
- **预测时域（N）**: 默认20步，涵盖1-2秒的预测时间
- **控制时域**: 默认10步，控制输入优化长度
- **时间步长（Δt）**: 默认0.1秒，可调整至0.05-0.2秒

#### 代价函数构成
数学表达式：
```
J = Σ[k=0,N] ||x_k - x_ref_k||²_Q + Σ[k=0,M] ||u_k||²_R + ||x_N - x_ref_N||²_Q_f + ΔJ_smooth
```

其中：
- **位置跟踪代价**: `||p_com - p_ref||²_Q_pos`, Q_pos = diag([100, 100])
- **速度跟踪代价**: `||v_com - v_ref||²_Q_vel`, Q_vel = diag([10, 10])
- **控制输入代价**: `||f_foot||²_R_force + ||p_foot||²_R_foot`
- **终端代价**: `||x_N - x_ref_N||²_Q_terminal`, 确保质心投影在支撑多边形内
- **平滑性代价**: `||Δf||²_R_rate + ||Δp_foot||²_R_rate`

### 2. 约束条件

#### 动力学约束
基于LIPM模型的离散化动力学：
```
x[k+1] = A*x[k] + B_force*f[k] + B_foot*p_foot[k]
```

其中状态向量 `x = [p_x, p_y, v_x, v_y]`

#### 物理约束
1. **足底力约束**:
   - 力大小限制: `||f|| ≤ f_max` (默认200N)
   - 摩擦锥约束: `||f_horizontal|| ≤ μ * f_vertical`

2. **质心运动约束**:
   - 位置界限: `p_min ≤ p_com ≤ p_max`
   - 速度界限: `v_min ≤ v_com ≤ v_max`

3. **落脚点约束**:
   - 步长限制: `||p_foot - p_foot_prev|| ≤ step_max`
   - 工作空间限制: 基于机器人运动学

## API使用说明

### 基本使用

```python
from gait_core.simplified_dynamics import SimplifiedDynamicsModel
from gait_core.mpc_controller import MPCController, MPCParameters
from gait_core.data_bus import get_data_bus, Vector3D

# 1. 创建LIPM模型
data_bus = get_data_bus()
lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)

# 2. 配置MPC参数
params = MPCParameters()
params.prediction_horizon = 20
params.control_horizon = 10
params.dt = 0.1

# 3. 创建MPC控制器
mpc_controller = MPCController(lipm_model, params)

# 4. 设置参考轨迹
target_velocity = Vector3D(x=0.3, y=0.0, z=0.0)
reference = mpc_controller.generate_walking_reference(
    target_velocity=target_velocity,
    step_length=0.2,
    step_width=0.15,
    step_duration=0.8
)
mpc_controller.set_reference_trajectory(reference)

# 5. 控制循环
dt = 0.1
for step in range(100):
    solution = mpc_controller.update_control(dt)
    if solution.success:
        print(f"步骤 {step}: 代价 {solution.cost:.2f}, 求解时间 {solution.solve_time:.3f}s")
```

### 高级配置

#### 控制模式选择
```python
from gait_core.mpc_controller import MPCMode

# 仅力控制
mpc_controller.control_mode = MPCMode.FORCE_CONTROL

# 仅步态控制
mpc_controller.control_mode = MPCMode.FOOTSTEP_CONTROL

# 力和步态联合控制
mpc_controller.control_mode = MPCMode.COMBINED_CONTROL
```

#### 权重调整
```python
import numpy as np

params = MPCParameters()

# 位置跟踪更重要
params.Q_position = np.diag([200.0, 200.0])

# 减少控制输入权重
params.R_force = np.diag([0.5, 0.5])

# 增加平滑性
params.R_force_rate = 10.0
```

#### 约束设置
```python
# 严格的力限制
params.max_force = 150.0
params.friction_coefficient = 0.6

# 工作空间限制
params.com_position_bounds = (-0.8, 0.8)
params.com_velocity_bounds = (-1.5, 1.5)

# 步态限制
params.max_step_length = 0.25
params.max_step_width = 0.2
```

## 优化求解器

### CVXPY求解器（推荐）
当安装了CVXPY时，使用凸优化求解器：
```bash
pip install cvxpy
```

优势：
- 更快的求解速度
- 更好的数值稳定性  
- 自动选择最优求解器（OSQP, ECOS等）

### SciPy求解器（后备）
当CVXPY不可用时，使用SLSQP求解器：
- 通用非线性优化
- 适用于简单问题
- 求解时间较长

## 性能指标

### 典型性能数据
| 预测时域 | 控制时域 | 求解时间 | 适用场景 |
|---------|---------|---------|----------|
| 10      | 5       | ~5ms    | 实时控制 |
| 20      | 10      | ~15ms   | 标准配置 |
| 30      | 15      | ~40ms   | 高精度预测 |

### 优化建议
1. **实时性要求高**: 减少预测时域至10-15步
2. **精度要求高**: 增加预测时域至25-30步
3. **步态切换频繁**: 使用COMBINED_CONTROL模式
4. **计算资源有限**: 使用FORCE_CONTROL模式

## 演示程序

### 运行基本演示
```bash
cd examples
python mpc_demo.py
```

演示内容：
1. **基本轨迹跟踪**: 正弦波参考轨迹跟踪
2. **行走控制**: 直线行走步态生成
3. **约束处理**: 严格约束下的控制性能

### 生成的图像
- `mpc_基本mpc跟踪.png`: 轨迹跟踪结果
- `mpc_walking.png`: 行走轨迹和落脚点
- `mpc_constraints.png`: 约束违反分析

## 数学原理

### LIPM动力学离散化
连续时间LIPM模型：
```
ẍ = (g/z_c)(x - p_foot) + f_x/m
ÿ = (g/z_c)(y - p_foot) + f_y/m
```

离散化（欧拉方法）：
```python
omega_n = sqrt(g/z_c)
A = [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [omega_n²*dt, 0, 1, 0],
     [0, omega_n²*dt, 0, 1]]

B_force = [[0, 0],
           [0, 0],
           [dt/m, 0],
           [0, dt/m]]
```

### ZMP（零力矩点）计算
ZMP位置计算：
```python
zmp_x = x - (z_c/g) * acceleration_x
zmp_y = y - (z_c/g) * acceleration_y
```

稳定性条件：ZMP应位于支撑多边形内

### 优化问题求解
使用二次规划（QP）求解：
```
minimize: (1/2) * x^T * H * x + f^T * x
subject to: A_eq * x = b_eq
            A_ineq * x <= b_ineq
```

## 故障诊断

### 常见问题

1. **求解失败**
   - 检查约束是否过于严格
   - 降低预测时域长度
   - 检查参考轨迹可行性

2. **求解时间过长**
   - 减少预测时域
   - 使用FORCE_CONTROL模式
   - 安装CVXPY优化求解

3. **轨迹跟踪误差大**
   - 增加位置跟踪权重
   - 检查LIPM参数准确性
   - 调整时间步长

4. **控制输入震荡**
   - 增加平滑性权重
   - 减少控制权重
   - 检查参考轨迹连续性

### 调试工具
```python
# 性能统计
stats = mpc_controller.get_performance_stats()
print(f"平均求解时间: {stats['average_solve_time']:.3f}s")

# 控制器状态
mpc_controller.print_status()

# 约束检查
from examples.mpc_demo import check_constraints
violations = check_constraints(state, force, params)
```

## 扩展开发

### 自定义代价函数
```python
class CustomMPCController(MPCController):
    def _evaluate_cost_scipy(self, x, current_state):
        # 基础代价
        base_cost = super()._evaluate_cost_scipy(x, current_state)
        
        # 添加自定义代价项
        custom_cost = self._compute_custom_cost(x, current_state)
        
        return base_cost + custom_cost
```

### 添加新约束
```python
def _solve_with_cvxpy(self, current_state):
    # ... 基础约束设置 ...
    
    # 添加自定义约束
    if self.enable_custom_constraints:
        constraints.extend(self._get_custom_constraints(states, forces))
    
    # ... 求解 ...
```

### 多目标优化
支持帕累托优化的多目标MPC：
```python
# 权重向量法
w1, w2 = 0.7, 0.3
cost = w1 * tracking_cost + w2 * energy_cost
```

## 参考文献

1. Kajita, S., et al. "Biped walking pattern generation by using preview control of zero-moment point." ICRA 2003.
2. Herdt, A., et al. "Walking without thinking about it." IROS 2010.  
3. Wieber, P. B. "Trajectory free linear model predictive control for stable walking in the presence of strong perturbations." Humanoids 2006.
4. Diedam, H., et al. "Online walking gait generation with adaptive foot positioning through linear model predictive control." IROS 2008.

## 更新日志

### v1.0 (当前版本)
- 基本MPC功能实现
- CVXPY和SciPy双求解器支持
- 三种控制模式（力/步态/组合）
- 完整的约束处理
- 性能监控和调试工具

### 计划功能
- 鲁棒MPC（处理模型不确定性）
- 分布式MPC（多足机器人）
- 自适应权重调整
- 机器学习辅助的参考轨迹生成 