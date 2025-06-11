# 足步规划系统 (Foot Placement Module)

## 概述

足步规划系统为AzureLoong人形机器人提供了智能的足步规划功能，能够根据当前支撑腿位置、步态参数和运动意图计算摆动腿的目标着地位置。

## 核心功能

### 1. 正向运动学计算
- **足端位置计算**: 从关节角度计算足端在世界坐标系中的位置
- **多关节链支持**: 支持髋关节、膝关节、踝关节的6自由度计算
- **双足同步计算**: 同时计算左右足的位置信息

### 2. 足步规划算法
- **相对位置规划**: 基于支撑腿位置计算摆动腿目标落脚点
- **步长步宽控制**: 
  - 标准步长: 0.15m (可调节 0.05-0.25m)
  - 标准步宽: 0.12m (可调节 0.08-0.20m)
  - 抬脚高度: 0.05m
- **动态速度适应**: 根据身体运动速度动态调整步长
- **稳定性保证**: 自动添加横向和纵向稳定性余量

### 3. 地形适应性
支持多种地形类型的适应性规划：
- **平地 (FLAT)**: 标准步态参数
- **斜坡 (SLOPE)**: 步长适当减小，步宽增加
- **楼梯 (STAIRS)**: 增加抬脚高度
- **粗糙地面 (ROUGH)**: 保守的步态参数

### 4. 规划策略
- **静态行走 (STATIC_WALK)**: 保守稳定的规划策略
- **动态行走 (DYNAMIC_WALK)**: 更快速的动态规划
- **自适应 (ADAPTIVE)**: 根据环境自动调整
- **地形自适应 (TERRAIN_ADAPTIVE)**: 特别针对复杂地形
- **稳定性优先 (STABILIZING)**: 最大化稳定性的规划

## 系统架构

### 核心组件

1. **FootPlacementPlanner**: 主要规划器类
   - 管理足部状态
   - 执行足步规划算法
   - 处理地形和策略适应

2. **ForwardKinematics**: 正向运动学计算器
   - 计算足端位置
   - 支持多关节链
   - 优化计算性能

3. **Vector3D**: 三维向量工具类
   - 位置表示和计算
   - 向量运算支持

4. **FootPlacementConfig**: 配置管理
   - 步态参数配置
   - 安全限制设置
   - 性能参数调优

### 数据总线集成

足步规划系统完全集成到数据总线中：

```python
# 触发足步规划
target_pos = data_bus.trigger_foot_placement_planning("left", "right")

# 设置运动意图
data_bus.set_body_motion_intent(forward_velocity=0.3, lateral_velocity=0.0)

# 获取目标位置
target = data_bus.get_target_foot_position("left")

# 获取足部位移
displacement = data_bus.get_foot_displacement("left")
distance = data_bus.get_foot_distance_to_target("left")
```

### 步态调度器集成

当步态状态发生转换时，系统会自动触发足步规划：

```python
# 状态转换时自动触发足步规划
state_changed = data_bus.update_gait_scheduler_with_foot_planning(dt)

# 或手动触发
if scheduler.swing_leg in ["left", "right"]:
    target_pos = scheduler._trigger_foot_placement_planning(old_state, new_state)
```

## 使用方法

### 基本使用

```python
from foot_placement import get_foot_planner, Vector3D

# 获取规划器实例
planner = get_foot_planner()

# 更新足部状态 (从关节角度)
joint_angles = {
    "left_hip_pitch": -0.1, "left_knee_pitch": 0.2, "left_ankle_pitch": -0.1,
    "right_hip_pitch": -0.1, "right_knee_pitch": 0.2, "right_ankle_pitch": -0.1
}
planner.update_foot_states_from_kinematics(joint_angles)

# 设置运动意图
planner.set_body_motion_intent(Vector3D(0.3, 0.0, 0.0), 0.0)  # 前进0.3m/s

# 执行足步规划
target_left = planner.plan_foot_placement("left", "right")   # 左腿摆动
target_right = planner.plan_foot_placement("right", "left")  # 右腿摆动

print(f"左腿目标: ({target_left.x:.3f}, {target_left.y:.3f}, {target_left.z:.3f})")
print(f"右腿目标: ({target_right.x:.3f}, {target_right.y:.3f}, {target_right.z:.3f})")
```

### 通过数据总线使用

```python
from data_bus import get_data_bus

data_bus = get_data_bus()

# 设置关节状态
data_bus.set_joint_position("left_hip_pitch", -0.1)
data_bus.set_joint_position("left_knee_pitch", 0.2)
# ... 其他关节

# 设置运动意图
data_bus.set_body_motion_intent(0.3, 0.0, 0.0)  # 前进0.3m/s

# 触发足步规划
target_pos = data_bus.trigger_foot_placement_planning("left", "right")

# 获取结果
if target_pos:
    print(f"规划成功: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")

# 查看详细状态
data_bus.print_foot_planning_status()
```

### 配置自定义参数

```python
from foot_placement import FootPlacementConfig, FootPlacementStrategy, TerrainType

# 创建自定义配置
config = FootPlacementConfig(
    nominal_step_length=0.18,      # 步长18cm
    nominal_step_width=0.14,       # 步宽14cm
    step_height=0.06,              # 抬脚高度6cm
    max_step_length=0.30,          # 最大步长30cm
    speed_step_gain=0.9            # 速度增益
)

# 使用自定义配置
planner = FootPlacementPlanner(config)

# 设置地形类型
planner.set_terrain_type(TerrainType.SLOPE)

# 设置规划策略
planner.set_planning_strategy(FootPlacementStrategy.DYNAMIC_WALK)
```

## 关键参数

### 步态参数
- `nominal_step_length`: 标准步长 [m] (默认: 0.15)
- `nominal_step_width`: 标准步宽 [m] (默认: 0.12)
- `step_height`: 抬脚高度 [m] (默认: 0.05)

### 身体几何参数
- `hip_width`: 髋宽 [m] (默认: 0.18)
- `leg_length`: 腿长 [m] (默认: 0.6)
- `foot_length`: 脚长 [m] (默认: 0.16)

### 动态调整参数
- `speed_step_gain`: 速度-步长增益 (默认: 0.8)
- `lateral_stability_margin`: 横向稳定性余量 [m] (默认: 0.03)
- `longitudinal_stability_margin`: 纵向稳定性余量 [m] (默认: 0.02)

### 安全限制
- `max_step_length`: 最大步长 [m] (默认: 0.25)
- `min_step_length`: 最小步长 [m] (默认: 0.05)
- `max_step_width`: 最大步宽 [m] (默认: 0.20)
- `min_step_width`: 最小步宽 [m] (默认: 0.08)

## 性能特征

### 计算性能
- **运动学计算**: >1000 Hz (约1ms/次)
- **足步规划**: >500 Hz (约2ms/次)
- **实时性能**: 支持50Hz+控制频率
- **计算复杂度**: O(1)时间复杂度

### 精度特征
- **位置精度**: ±1mm
- **运动学精度**: 基于简化DH参数，适合实时控制
- **规划一致性**: 相同输入产生相同输出

## 测试验证

### 运行测试
```bash
# 激活虚拟环境
source venv/bin/activate

# 基本功能测试
python simple_test.py

# 完整系统演示
python gait_with_foot_planning_demo.py

# 单独模块测试
python foot_placement.py
```

### 测试覆盖
- ✅ 正向运动学计算
- ✅ 基本足步规划
- ✅ 数据总线集成
- ✅ 步态调度器集成
- ✅ 地形适应性
- ✅ 多种规划策略
- ✅ 性能基准测试

## 技术特点

### 优势
1. **实时性能**: 高频率计算支持，满足机器人控制需求
2. **集成性好**: 与现有数据总线和步态调度器无缝集成
3. **可配置性**: 丰富的配置选项，适应不同需求
4. **稳定性**: 内置稳定性保证机制
5. **扩展性**: 模块化设计，易于扩展新功能

### 应用场景
- **平地行走**: 标准步态规划
- **复杂地形**: 自适应地形规划
- **变速行走**: 动态速度适应
- **精确定位**: 高精度足端控制
- **稳定性控制**: 增强行走稳定性

## 未来扩展

### 计划功能
- [ ] 障碍物避让规划
- [ ] 多步预测规划
- [ ] 学习式参数优化
- [ ] 视觉反馈集成
- [ ] 力反馈优化

### 性能优化
- [ ] GPU加速计算
- [ ] 更精确的运动学模型
- [ ] 自适应参数调节
- [ ] 预测性规划算法

## 维护说明

### 依赖项
- `numpy`: 数值计算
- `threading`: 线程安全
- `dataclasses`: 数据结构
- `enum`: 枚举类型

### 文件结构
```
foot_placement.py          # 主要足步规划模块
data_bus.py                # 数据总线集成
gait_scheduler.py          # 步态调度器集成
simple_test.py             # 基本测试脚本
gait_with_foot_planning_demo.py  # 完整演示脚本
README_FootPlacement.md    # 本文档
```

### 版本信息
- **当前版本**: 1.0
- **兼容性**: Python 3.7+
- **最后更新**: 2024年12月

---

**作者**: Adam Control Team  
**联系**: 技术支持团队  
**许可**: MIT License 