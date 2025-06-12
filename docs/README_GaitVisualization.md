# 步态可视化仿真使用说明

## 功能概述

步态可视化仿真脚本（`gait_visualization_simulation.py`）提供了一个实时的3D仿真环境，用于可视化机器人的步态状态和足步规划结果。

### 主要功能

#### 1. 实时步态状态显示
- **颜色标注**: 机器人躯干颜色根据当前步态状态变化
  - 🟢 **绿色**: 左腿支撑（RIGHT_SUPPORT）
  - 🔵 **蓝色**: 右腿支撑（LEFT_SUPPORT）  
  - 🟡 **黄色**: 双支撑LR（DOUBLE_SUPPORT_LR）
  - 🟠 **橙色**: 双支撑RL（DOUBLE_SUPPORT_RL）
  - ⚫ **灰色**: 站立状态（STANDING）
  - ⚫ **深灰**: 空闲状态（IDLE）

- **文字信息**: 左上角显示详细状态信息
  - 当前步态状态
  - 摆动腿/支撑腿标识
  - 足底力读数
  - 目标足部位置坐标

#### 2. 足部目标位置可视化
- **红色小球**: 左脚目标位置标记
- **绿色小球**: 右脚目标位置标记
- **实时更新**: 根据足步规划算法实时移动位置

#### 3. 集成的步态系统
- **步态调度器**: 自动状态机转换
- **足步规划器**: 智能目标位置计算
- **传感器仿真**: 基于足部高度的接触力计算

## 运行方式

### GUI模式（推荐）
```bash
# 使用默认模型
python gait_visualization_simulation.py

# 指定模型文件
python gait_visualization_simulation.py --model models/AzureLoong.xml
```

### 无头模式（测试用）
```bash
# 运行5秒无头测试
python gait_visualization_simulation.py --headless --duration 5

# 运行10秒无头测试
python gait_visualization_simulation.py --headless --duration 10
```

## 交互控制

### 键盘控制
| 按键 | 功能 |
|------|------|
| **空格键** | 暂停/继续仿真 |
| **R** | 重置仿真状态 |
| **+/=** | 加速仿真（最高5倍速） |
| **-** | 减速仿真（最低0.1倍速） |
| **ESC** | 退出仿真 |

### 鼠标控制
- **拖拽**: 旋转视角
- **滚轮**: 缩放视图
- **右键拖拽**: 平移视图

## 系统架构

### 核心组件
```
步态可视化仿真
├── MuJoCo物理引擎
│   ├── 双足机器人模型
│   ├── 物理仿真计算
│   └── 可视化渲染
├── 步态系统集成
│   ├── DataBus数据总线
│   ├── GaitScheduler步态调度器
│   └── FootPlacementPlanner足步规划器
└── 可视化组件
    ├── 状态颜色映射
    ├── 目标位置标记
    └── 文本信息覆盖
```

### 数据流
```
传感器数据 → DataBus → GaitScheduler → 状态更新
     ↓                      ↓
 足部接触力              步态状态
     ↓                      ↓
可视化更新 ← FootPlacementPlanner ← 足步规划
```

## 可视化元素详解

### 1. 机器人模型
- **躯干**: 中央灰色方块，颜色表示步态状态
- **腿部**: 胶囊形状的大腿和小腿
- **足部**: 矩形足部，检测地面接触

### 2. 目标位置标记
- **半透明球体**: 不参与物理碰撞
- **动态更新**: 跟随足步规划算法移动
- **颜色区分**: 左脚红色，右脚绿色

### 3. 状态信息面板
```
步态状态可视化
────────────────────
当前状态: left_support
摆动腿: right
支撑腿: left
────────────────────
足底力:
  左脚: 80.5N
  右脚: 0.0N
────────────────────
目标位置:
right_foot: (0.27, 0.18, 0.02)
────────────────────
控制:
  空格键: 暂停/继续
  R: 重置
  +/-: 调节速度
  ESC: 退出
```

## 技术特性

### 仿真参数
- **控制频率**: 1000Hz（1ms控制周期）
- **物理仿真**: MuJoCo 3.x引擎
- **可视化**: 实时3D渲染
- **响应时间**: <1ms状态更新

### 步态参数
- **摆动时间**: 400ms
- **双支撑时间**: 100ms
- **接触力阈值**: 30N
- **步长**: 可配置

### 控制算法
- **PD控制器**: 关节位置控制
- **平衡控制**: 简化的直立保持
- **接触检测**: 基于足部高度

## 故障排除

### 常见问题

1. **模型文件未找到**
   ```
   解决方案: 脚本会自动使用内置简单双足模型
   ```

2. **MuJoCo未安装**
   ```bash
   pip install mujoco
   ```

3. **显示器问题（无头服务器）**
   ```bash
   # 使用无头模式
   python gait_visualization_simulation.py --headless
   ```

4. **性能问题**
   ```
   解决方案: 降低仿真速度或关闭复杂渲染
   ```

### 调试选项

```bash
# 启用详细日志
python gait_visualization_simulation.py --verbose

# 指定不同的物理引擎参数
python gait_visualization_simulation.py --dt 0.002
```

## 开发和扩展

### 添加新的可视化元素
```python
# 在_update_visualization方法中添加
def _update_custom_markers(self):
    # 自定义可视化逻辑
    pass
```

### 修改状态颜色
```python
self.leg_state_colors = {
    GaitState.LEFT_SUPPORT: [r, g, b, a],  # 自定义颜色
    # ...
}
```

### 集成新的传感器
```python
def _read_custom_sensors(self):
    # 读取额外传感器数据
    # 更新DataBus
    pass
```

## 性能优化建议

1. **降低控制频率**: 对于可视化不需要1000Hz
2. **简化模型**: 使用更简单的几何体
3. **减少标记数量**: 只显示必要的目标位置
4. **优化渲染**: 降低渲染质量设置

## 相关文件

- `gait_visualization_simulation.py`: 主仿真脚本
- `data_bus.py`: 数据总线
- `gait_scheduler.py`: 步态调度器
- `foot_placement.py`: 足步规划器
- `models/`: 机器人模型文件

## 版本历史

- v1.0: 基础可视化功能
- v1.1: 添加交互控制
- v1.2: 集成完整步态系统
- v1.3: 性能优化和错误处理

---

**作者**: Adam Control Team  
**更新日期**: 2024年12月  
**版本**: 1.3 