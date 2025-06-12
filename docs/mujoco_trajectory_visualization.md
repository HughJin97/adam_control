# MuJoCo轨迹可视化系统

## 概述

本系统为AzureLoong双足机器人提供了完整的足端轨迹可视化解决方案，支持在MuJoCo仿真环境中实时显示和分析足端轨迹规划。

## 系统特性

### 🎯 核心功能
- **实时轨迹可视化**: 在MuJoCo环境中实时显示足端轨迹
- **多种轨迹类型**: 支持多项式、贝塞尔曲线等轨迹规划算法
- **相位进度显示**: 实时显示轨迹执行的相位进度
- **预测轨迹**: 显示未来轨迹路径预测
- **目标点可视化**: 清晰显示足端目标位置
- **历史轨迹记录**: 保存和显示历史轨迹路径

### 🎨 可视化元素
- **绿色轨迹**: 左足历史轨迹路径
- **红色轨迹**: 右足历史轨迹路径
- **蓝色点**: 预测轨迹点（渐变透明度）
- **黄色球**: 足端目标位置
- **绿/红小球**: 相位进度指示器（高度表示进度）

### 🔧 技术特性
- **跨平台兼容**: 支持macOS/Linux/Windows
- **自动降级**: macOS上自动切换到无GUI模式
- **数据记录**: 自动保存轨迹数据为JSON格式
- **详细分析**: 生成速度、相位、3D轨迹等分析图表

## 文件结构

```
adam_control/
├── examples/
│   ├── mujoco_trajectory_visualization.py  # 主要可视化脚本
│   └── plot_trajectory_data.py            # 轨迹数据分析脚本
├── run_trajectory_visualization.py        # 便捷启动脚本
├── models/
│   ├── scene.xml                         # MuJoCo场景文件
│   └── AzureLoong.xml                    # 机器人模型文件
└── docs/
    └── mujoco_trajectory_visualization.md # 本文档
```

## 快速开始

### 1. 环境要求

```bash
# 安装必要的Python包
pip install mujoco numpy matplotlib

# 可选：安装mjpython（macOS推荐）
# 从MuJoCo官网下载并安装
```

### 2. 运行轨迹可视化

```bash
# 方法1：使用便捷启动脚本
python run_trajectory_visualization.py

# 方法2：直接运行可视化脚本
python examples/mujoco_trajectory_visualization.py

# 方法3：在macOS上使用mjpython（推荐）
mjpython run_trajectory_visualization.py
```

### 3. 分析轨迹数据

```bash
# 生成详细的轨迹分析图表
python examples/plot_trajectory_data.py
```

## 使用说明

### 交互控制（GUI模式）

当成功启动MuJoCo viewer时，可以使用以下控制：

- **鼠标拖拽**: 旋转视角
- **鼠标滚轮**: 缩放视图
- **右键拖拽**: 平移视图
- **ESC键**: 退出程序

### 无GUI模式

在macOS等不支持MuJoCo viewer的环境中，系统会自动切换到无GUI模式：

- 运行10秒的轨迹仿真
- 每秒输出状态信息
- 自动保存轨迹数据
- 生成详细的统计报告

### 输出文件

#### 轨迹数据文件
- **文件名**: `trajectory_data_<timestamp>.json`
- **内容**: 包含时间序列、位置、相位、状态等完整数据
- **格式**: JSON格式，便于后续分析

#### 分析图表
- **3D轨迹图**: `*_3d_trajectory.png`
  - 3D足端轨迹
  - XY平面投影
  - 高度变化曲线
  - 相位变化曲线

- **速度分析图**: `*_velocity_analysis.png`
  - X/Y/Z方向速度
  - 速度大小变化

- **相位分析图**: `*_phase_analysis.png`
  - 相位变化曲线
  - 左右足相位差
  - 状态时间线

## 配置参数

### 轨迹配置

```python
# 左足配置
left_config = TrajectoryConfig(
    step_height=0.08,              # 抬脚高度 [m]
    swing_duration=1.0,            # 摆动时间 [s]
    interpolation_type="cubic",    # 插值类型
    vertical_trajectory_type="sine" # 垂直轨迹类型
)

# 右足配置
right_config = TrajectoryConfig(
    step_height=0.06,              # 抬脚高度 [m]
    swing_duration=1.2,            # 摆动时间 [s]
    interpolation_type="cubic",    # 插值类型
    vertical_trajectory_type="sine" # 垂直轨迹类型
)
```

### 仿真参数

```python
# 仿真设置
dt = 0.002                    # 时间步长 [s]
max_sim_time = 10.0          # 最大仿真时间 [s]
trajectory_history_length = 200  # 历史轨迹点数
```

### 可视化参数

```python
# 颜色配置
trajectory_colors = {
    'left_foot': [0, 1, 0, 0.8],    # 绿色
    'right_foot': [1, 0, 0, 0.8],   # 红色
    'predicted': [0, 0, 1, 0.6],    # 蓝色
    'target': [1, 1, 0, 1.0]        # 黄色
}
```

## 技术架构

### 核心组件

1. **MuJoCoTrajectoryVisualizer**: 主要可视化类
   - 管理MuJoCo仿真环境
   - 处理轨迹规划器
   - 实现可视化渲染

2. **SimpleDataBusAdapter**: 数据总线适配器
   - 提供统一的数据接口
   - 兼容FootTrajectory类
   - 支持数据更新和查询

3. **FootTrajectory**: 足端轨迹规划器
   - 实现多种轨迹算法
   - 支持实时更新
   - 提供状态管理

### 数据流

```
MuJoCo仿真 → 足端位置 → 轨迹规划器 → 可视化渲染
     ↓              ↓           ↓
传感器数据 → 数据总线 → 状态更新 → 数据记录
```

## 故障排除

### 常见问题

#### 1. MuJoCo viewer启动失败
**错误**: `launch_passive requires that the Python script be run under mjpython on macOS`

**解决方案**:
- 安装mjpython: 从MuJoCo官网下载
- 使用mjpython运行: `mjpython run_trajectory_visualization.py`
- 或者使用无GUI模式（自动切换）

#### 2. 中文字体显示问题
**现象**: matplotlib图表中中文显示为方框

**解决方案**:
```python
# 安装中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

#### 3. 轨迹数据文件未生成
**检查**:
- 确认程序运行完整
- 检查当前目录写权限
- 查看控制台错误信息

#### 4. 图表生成失败
**可能原因**:
- matplotlib版本不兼容
- 数据格式错误
- 内存不足

**解决方案**:
```bash
# 更新matplotlib
pip install --upgrade matplotlib

# 检查数据文件完整性
python -c "import json; print(json.load(open('trajectory_data_*.json')))"
```

## 扩展开发

### 添加新的轨迹类型

```python
# 在TrajectoryConfig中添加新类型
class TrajectoryConfig:
    def __init__(self):
        self.trajectory_type = "custom"  # 新类型
        # ... 其他参数

# 在FootTrajectory中实现新算法
def custom_trajectory(self, phase):
    # 实现自定义轨迹算法
    pass
```

### 自定义可视化元素

```python
def add_custom_visualization(self, viewer):
    """添加自定义可视化元素"""
    # 添加新的标记
    viewer.add_marker(
        pos=custom_position,
        size=[0.01, 0.01, 0.01],
        rgba=[1, 0, 1, 1],  # 紫色
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        label="custom_marker"
    )
```

### 数据分析扩展

```python
def custom_analysis(data):
    """自定义数据分析"""
    # 计算自定义指标
    custom_metric = calculate_custom_metric(data)
    
    # 生成自定义图表
    plt.figure()
    plt.plot(data['time'], custom_metric)
    plt.title('Custom Analysis')
    plt.show()
```

## 性能优化

### 实时性能
- 使用合适的时间步长（推荐0.002s）
- 限制历史轨迹点数（推荐200点）
- 优化可视化更新频率

### 内存使用
- 定期清理历史数据
- 使用deque限制数据长度
- 避免大量numpy数组复制

### 渲染性能
- 减少可视化元素数量
- 使用适当的透明度
- 优化标记大小和密度

## 版本历史

### v1.0.0 (当前版本)
- 基础轨迹可视化功能
- MuJoCo集成
- 数据记录和分析
- 跨平台兼容性

### 计划功能
- 实时参数调整界面
- 多机器人支持
- 轨迹对比分析
- 性能基准测试

## 贡献指南

欢迎提交问题报告和功能请求！

### 开发环境设置
```bash
# 克隆项目
git clone <repository_url>
cd adam_control

# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/
```

### 代码规范
- 遵循PEP 8编码规范
- 添加适当的文档字符串
- 编写单元测试
- 提交前运行linter检查

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系开发团队或提交GitHub Issue。 