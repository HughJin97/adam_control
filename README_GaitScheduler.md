# 步态调度器 (Gait Scheduler) 使用指南

## 概述

步态调度器是AzureLoong机器人控制系统的核心组件，实现了基于有限状态机的步态管理。它能够根据时间和传感器反馈自动管理机器人的步态状态转换，支持多种步态模式和安全机制。

## 核心功能

### 🎯 主要特性

1. **有限状态机管理** - 实现完整的步态状态转换逻辑
2. **混合触发机制** - 支持基于时间和传感器的状态转换
3. **数据总线集成** - 与机器人控制系统无缝集成
4. **实时监控** - 提供状态统计和转换历史
5. **安全机制** - 紧急停止和异常处理
6. **配置灵活** - 动态调整步态参数

### 📊 支持的步态状态

| 状态 | 代码 | 描述 | 支撑腿状态 |
|------|------|------|------------|
| IDLE | `idle` | 空闲状态 | DSt (双支撑) |
| STANDING | `standing` | 静止站立 | DSt (双支撑) |
| LEFT_SUPPORT | `left_support` | 左腿支撑，右腿摆动 | LSt (左支撑) |
| RIGHT_SUPPORT | `right_support` | 右腿支撑，左腿摆动 | RSt (右支撑) |
| DOUBLE_SUPPORT_LR | `double_support_lr` | 双支撑过渡（左→右） | DSt (双支撑) |
| DOUBLE_SUPPORT_RL | `double_support_rl` | 双支撑过渡（右→左） | DSt (双支撑) |
| FLIGHT | `flight` | 飞行相（跑步时） | NSt (无支撑) |
| EMERGENCY_STOP | `emergency_stop` | 紧急停止 | DSt (双支撑) |

## 快速开始

### 基本使用

```python
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig
from data_bus import get_data_bus
import numpy as np

# 1. 初始化调度器
config = GaitSchedulerConfig(
    swing_time=0.4,              # 摆动时间
    stance_time=0.6,             # 支撑时间
    double_support_time=0.1,     # 双支撑时间
    touchdown_force_threshold=30.0,  # 着地力阈值
    liftoff_force_threshold=10.0     # 离地力阈值
)

scheduler = get_gait_scheduler(config)
data_bus = get_data_bus()

# 2. 开始行走
data_bus.start_gait_scheduler_walking()

# 3. 主控制循环
while True:
    # 更新传感器数据 (来自实际传感器或仿真)
    left_force = 60.0  # 左脚力 [N]
    right_force = 30.0  # 右脚力 [N]
    left_velocity = np.array([0.0, 0.0, 0.01])   # 左脚速度 [m/s]
    right_velocity = np.array([0.0, 0.0, 0.05])  # 右脚速度 [m/s]
    
    # 更新调度器
    dt = 0.02  # 时间步长 [s]
    state_changed = data_bus.update_gait_scheduler(dt)
    
    if state_changed:
        print(f"状态转换到: {data_bus.current_gait_state}")
        print(f"支撑腿: {data_bus.get_current_support_leg()}")
        print(f"摆动腿: {data_bus.get_current_swing_leg()}")
    
    # 检查步态状态
    if data_bus.is_in_swing_phase('left'):
        print("左腿在摆动相")
    
    if data_bus.is_in_double_support():
        print("处于双支撑相")
    
    time.sleep(dt)
```

### 高级配置

```python
# 高级配置示例
advanced_config = GaitSchedulerConfig(
    # 时间参数
    swing_time=0.35,                    # 更快的摆动
    stance_time=0.55,                   # 相应的支撑时间
    double_support_time=0.08,           # 较短的双支撑
    
    # 传感器阈值
    touchdown_force_threshold=25.0,      # 更敏感的着地检测
    liftoff_force_threshold=8.0,        # 更敏感的离地检测
    contact_velocity_threshold=0.01,     # 更严格的速度阈值
    
    # 触发模式
    use_time_trigger=True,              # 启用时间触发
    use_sensor_trigger=True,            # 启用传感器触发
    require_both_triggers=False,        # OR 逻辑（任一满足即触发）
    
    # 安全参数
    max_swing_time=0.8,                 # 最大摆动时间限制
    min_stance_time=0.15,               # 最小支撑时间限制
    emergency_force_threshold=150.0,     # 紧急停止力阈值
    
    # 调试选项
    enable_logging=True,                # 启用状态转换日志
    log_transitions=True,               # 记录转换历史
    max_log_entries=500                 # 日志条目数限制
)
```

## 状态转换逻辑

### 🔄 转换触发条件

#### 1. 基于时间的转换
- **摆动相结束**: 经过 `swing_time` 后触发转换
- **双支撑结束**: 经过 `double_support_time` 后触发转换

#### 2. 基于传感器的转换
- **摆动脚着地**: 摆动脚力 > `touchdown_force_threshold` 且速度 < `contact_velocity_threshold`
- **支撑脚离地**: 支撑脚力 < `liftoff_force_threshold`
- **紧急情况**: 任一脚力 > `emergency_force_threshold`

#### 3. 混合触发模式
```python
# OR 逻辑 (默认)
require_both_triggers = False  # 时间 OR 传感器

# AND 逻辑 (更严格)
require_both_triggers = True   # 时间 AND 传感器
```

### 📈 典型步态循环

```
STANDING → LEFT_SUPPORT → DOUBLE_SUPPORT_LR → RIGHT_SUPPORT → DOUBLE_SUPPORT_RL → LEFT_SUPPORT → ...
```

**详细时序**:
1. **LEFT_SUPPORT** (0.4s): 左腿支撑，右腿摆动
2. **DOUBLE_SUPPORT_LR** (0.1s): 双腿支撑，准备切换
3. **RIGHT_SUPPORT** (0.4s): 右腿支撑，左腿摆动  
4. **DOUBLE_SUPPORT_RL** (0.1s): 双腿支撑，准备切换
5. **循环继续**...

## 数据总线集成

### 🔗 状态字段

调度器会自动更新数据总线中的以下字段：

```python
# 基本状态
data_bus.legState                    # "LSt"/"RSt"/"DSt"/"NSt"
data_bus.current_gait_state          # 当前步态状态名称
data_bus.swing_leg                   # "left"/"right"/"none"/"both"
data_bus.support_leg                 # "left"/"right"/"both"/"none"

# 统计信息
data_bus.state_transition_count      # 状态转换次数
data_bus.last_state_transition_time  # 上次转换时间

# 步态数据结构更新
data_bus.gait.left_in_swing          # 左腿是否在摆动相
data_bus.gait.right_in_swing         # 右腿是否在摆动相
data_bus.gait.double_support         # 是否在双支撑相
```

### 🛠 接口方法

#### 状态更新
```python
# 更新调度器 (主要接口)
state_changed = data_bus.update_gait_scheduler(dt)

# 同步状态到数据总线 (内部调用)
data_bus._sync_scheduler_state()
```

#### 控制接口
```python
# 启动/停止
data_bus.start_gait_scheduler_walking()
data_bus.stop_gait_scheduler_walking()
data_bus.emergency_stop_gait()
data_bus.reset_gait_scheduler()

# 配置管理
config_dict = {'swing_time': 0.3, 'touchdown_force_threshold': 25.0}
data_bus.set_gait_scheduler_config(config_dict)
```

#### 查询接口
```python
# 状态查询
current_support = data_bus.get_current_support_leg()      # "left"/"right"/"both"/"none"
current_swing = data_bus.get_current_swing_leg()          # "left"/"right"/"none"/"both"

# 相位检查
is_left_swing = data_bus.is_in_swing_phase('left')        # True/False
is_right_support = data_bus.is_in_support_phase('right')  # True/False
is_double_support = data_bus.is_in_double_support()       # True/False

# 状态信息
state_info = data_bus.get_gait_state_info()              # 详细状态字典
statistics = data_bus.get_gait_state_statistics()        # 统计信息
```

## 监控和调试

### 📊 状态监控

```python
# 实时状态打印
data_bus.print_gait_status()

# 输出示例:
"""
=== 步态状态总览 ===
支撑腿状态: LSt
当前步态状态: left_support
摆动腿: right
支撑腿: left
状态转换次数: 15
当前状态持续时间: 0.245s
调度器总运行时间: 8.350s
左脚接触力: 65.2N
右脚接触力: 8.1N
左脚接触: True
右脚接触: False
==============================
"""
```

### 📈 统计分析

```python
# 获取详细统计
stats = data_bus.get_gait_state_statistics()

print(f"总运行时间: {stats['total_duration']:.2f}s")
print(f"总转换次数: {stats['total_transitions']}")

# 各状态时间分布
for state, data in stats['state_stats'].items():
    print(f"{state}: {data['percentage']:.1f}% "
          f"(平均 {data['avg_duration']:.3f}s)")
```

### 🔍 回调机制

```python
# 添加状态变化回调
def on_gait_state_change(old_state, new_state, duration):
    print(f"[步态] {old_state.value} → {new_state.value} ({duration:.3f}s)")
    
    # 自定义逻辑
    if new_state == GaitState.LEFT_SUPPORT:
        print("开始右腿摆动，调整右腿控制器")
    elif new_state == GaitState.DOUBLE_SUPPORT_LR:
        print("进入双支撑，准备重心转移")

data_bus.add_gait_state_callback(on_gait_state_change)
```

## 安全机制

### ⚠️ 紧急停止

**自动触发条件**:
- 足部力超过紧急阈值
- 传感器异常读数
- 状态机异常

**手动触发**:
```python
# 立即紧急停止
data_bus.emergency_stop_gait()

# 检查紧急状态
if scheduler.current_state == GaitState.EMERGENCY_STOP:
    print("系统处于紧急停止状态")
    
    # 恢复流程
    data_bus.reset_gait_scheduler()
    data_bus.start_gait_scheduler_walking()
```

### 🛡️ 参数限制

```python
# 内置安全限制
config = GaitSchedulerConfig(
    max_swing_time=1.0,          # 防止摆动过长
    min_stance_time=0.2,         # 确保最小支撑时间
    emergency_force_threshold=200.0  # 紧急力阈值
)
```

## 测试和验证

### 🧪 运行测试

```bash
# 完整功能测试
python test_gait_scheduler.py

# 基础演示
python gait_scheduler_demo.py basic

# 动态参数调整演示
python gait_scheduler_demo.py dynamic

# 紧急场景演示
python gait_scheduler_demo.py emergency

# 传感器融合演示
python gait_scheduler_demo.py fusion
```

### 📋 测试覆盖

- ✅ 基本状态转换
- ✅ 传感器驱动转换
- ✅ 混合触发模式
- ✅ 数据总线集成
- ✅ 紧急停止机制
- ✅ 配置动态调整
- ✅ 统计和监控
- ✅ 可视化图表生成

## 性能优化

### ⚡ 实时性能

- **控制频率**: 支持50Hz+ 实时控制
- **线程安全**: 使用RLock确保多线程安全
- **内存效率**: 有限的历史记录存储
- **计算复杂度**: O(1) 状态更新复杂度

### 🎯 调优建议

```python
# 高频控制优化
config = GaitSchedulerConfig(
    enable_logging=False,        # 关闭详细日志
    log_transitions=False,       # 关闭转换记录
    max_log_entries=100         # 减少历史记录
)

# 传感器噪声处理
config.touchdown_force_threshold = actual_threshold + noise_margin
config.contact_velocity_threshold = actual_threshold + velocity_margin
```

## 故障排除

### ❓ 常见问题

**Q: 状态转换不正常？**
A: 检查传感器数据质量和阈值设置
```python
# 调试传感器数据
info = data_bus.get_gait_state_info()
print(f"左脚力: {info['left_foot_force']:.1f}N")
print(f"右脚力: {info['right_foot_force']:.1f}N")
print(f"阈值: {scheduler.config.touchdown_force_threshold:.1f}N")
```

**Q: 步态过快或过慢？**
A: 调整时间参数
```python
# 调整步态速度
config_update = {
    'swing_time': 0.3,           # 减少摆动时间 = 加快步频
    'double_support_time': 0.08  # 减少双支撑时间
}
data_bus.set_gait_scheduler_config(config_update)
```

**Q: 传感器触发不灵敏？**
A: 调整传感器阈值
```python
sensitive_config = {
    'touchdown_force_threshold': 20.0,  # 降低阈值 = 更敏感
    'contact_velocity_threshold': 0.01  # 降低速度阈值
}
data_bus.set_gait_scheduler_config(sensitive_config)
```

### 🔧 调试工具

```python
# 启用详细日志
scheduler.config.enable_logging = True
scheduler.config.log_transitions = True

# 打印转换历史
for log_entry in scheduler.transition_log:
    print(f"{log_entry['timestamp']:.2f}: "
          f"{log_entry['from_state']} → {log_entry['to_state']}")

# 实时状态监控
scheduler.print_status()  # 调度器状态
data_bus.print_gait_status()  # 数据总线状态
```

## 扩展开发

### 🔧 自定义状态

```python
from gait_scheduler import GaitState, StateTransition, TransitionTrigger

# 添加自定义状态（需要修改枚举）
# 添加自定义转换
def custom_condition(scheduler):
    # 自定义转换条件逻辑
    return custom_logic_result

scheduler.add_transition(
    GaitState.CUSTOM_STATE_1,
    GaitState.CUSTOM_STATE_2,
    TransitionTrigger.HYBRID,
    condition_func=custom_condition
)
```

### 🎯 自定义触发器

```python
# 实现自定义传感器融合
class CustomGaitScheduler(GaitScheduler):
    def _check_sensor_condition(self, transition):
        # 重写传感器条件检查
        result = super()._check_sensor_condition(transition)
        
        # 添加自定义逻辑
        if self.use_custom_sensors:
            result = result and self.check_custom_sensors()
        
        return result
```

## 版本信息

- **当前版本**: 1.0
- **兼容性**: Python 3.7+
- **依赖项**: numpy, threading, dataclasses, enum
- **可选依赖**: matplotlib (用于可视化)

## 更新日志

### v1.0 (当前版本)
- ✅ 实现完整的步态有限状态机
- ✅ 混合触发机制 (时间 + 传感器)
- ✅ 数据总线深度集成
- ✅ 实时监控和统计
- ✅ 安全机制和异常处理
- ✅ 配置灵活性和动态调整
- ✅ 全面的测试覆盖
- ✅ 详细的使用文档

---

> **注意**: 这是一个核心控制组件，建议在实际部署前进行充分测试。如有问题，请查看测试脚本或参考故障排除章节。 