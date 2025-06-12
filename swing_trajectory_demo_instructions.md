
# MuJoCo轨迹可视化使用说明

## 生成的文件
- swing_trajectory_demo.xml: 包含轨迹可视化的MuJoCo模型文件

## 查看方法

### 方法1: MuJoCo应用程序（推荐）
1. 下载并安装MuJoCo: https://mujoco.org/
2. 打开MuJoCo应用程序
3. 拖拽swing_trajectory_demo.xml到MuJoCo窗口中
4. 或者使用菜单: File -> Load Model -> 选择swing_trajectory_demo.xml

### 方法2: MjPython命令行
```bash
mjpython -m mujoco.viewer --mjcf swing_trajectory_demo.xml
```

### 方法3: Python脚本
```python
import mujoco as mj
import mujoco.viewer as viewer

model = mj.MjModel.from_xml_path("swing_trajectory_demo.xml")
data = mj.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mj.mj_step(model, data)
        v.sync()
```

## 轨迹可视化说明
- 红色半透明线条: 左脚轨迹
- 蓝色半透明线条: 右脚轨迹
- 蓝色球体: 轨迹起点
- 橙色球体: 轨迹终点
- 紫色球体: 轨迹峰值点

## 操作控制
- 鼠标左键拖拽: 旋转视角
- 鼠标右键拖拽: 平移视角
- 滚轮: 缩放
- 空格键: 暂停/继续仿真
- R键: 重置仿真
