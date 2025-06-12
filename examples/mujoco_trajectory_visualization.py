#!/usr/bin/env python3
"""
MuJoCo轨迹可视化演示
在MuJoCo仿真环境中显示AzureLoong机器人的足端轨迹规划和实时跟踪
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os
from collections import deque

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gait_core.foot_trajectory import FootTrajectory, TrajectoryConfig
from gait_core.data_bus import DataBus

class SimpleDataBusAdapter:
    """简单的数据总线适配器，提供update和get方法"""
    
    def __init__(self):
        self.data = {}
        self.real_data_bus = DataBus()
    
    def update(self, key: str, value):
        """更新数据"""
        self.data[key] = value
    
    def get(self, key: str):
        """获取数据"""
        return self.data.get(key, None)
    
    def get_foot_target(self, foot_name: str):
        """获取足部目标位置（兼容FootTrajectory接口）"""
        if foot_name == 'LF':
            return self.get('left_foot_target')
        elif foot_name == 'RF':
            return self.get('right_foot_target')
        return None
    
    def get_foot_position(self, foot_name: str):
        """获取足部当前位置（兼容FootTrajectory接口）"""
        if foot_name == 'LF':
            return self.get('left_foot_position')
        elif foot_name == 'RF':
            return self.get('right_foot_position')
        return None

class MuJoCoTrajectoryVisualizer:
    """MuJoCo轨迹可视化器"""
    
    def __init__(self, model_path="models/scene.xml"):
        """初始化可视化器"""
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        
        # 轨迹历史记录
        self.trajectory_history = {
            'left_foot': deque(maxlen=200),   # 保存200个历史点
            'right_foot': deque(maxlen=200)
        }
        
        # 预测轨迹点
        self.predicted_trajectory = {
            'left_foot': [],
            'right_foot': []
        }
        
        # 足端轨迹规划器
        self.data_bus = SimpleDataBusAdapter()
        self.left_foot_traj = None
        self.right_foot_traj = None
        
        # 简单的数据存储字典（用于轨迹规划）
        self.trajectory_data = {
            'left_foot_position': np.zeros(3),
            'right_foot_position': np.zeros(3),
            'left_foot_target': np.zeros(3),
            'right_foot_target': np.zeros(3),
            'left_foot_contact_force': 0.0,
            'right_foot_contact_force': 0.0
        }
        
        # 仿真参数
        self.dt = 0.002  # 2ms时间步长
        self.sim_time = 0.0
        
        # 可视化参数
        self.trajectory_colors = {
            'left_foot': [0, 1, 0, 0.8],    # 绿色
            'right_foot': [1, 0, 0, 0.8],   # 红色
            'predicted': [0, 0, 1, 0.6],    # 蓝色
            'target': [1, 1, 0, 1.0]        # 黄色
        }
        
        # 轨迹可视化对象ID
        self.trajectory_geom_ids = []
        
    def initialize_mujoco(self):
        """初始化MuJoCo仿真环境"""
        try:
            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # 设置初始姿态
            self.set_initial_pose()
            
            # 前向运动学计算
            mujoco.mj_forward(self.model, self.data)
            
            print(f"✓ MuJoCo模型加载成功: {self.model_path}")
            print(f"  - 自由度数量: {self.model.nq}")
            print(f"  - 执行器数量: {self.model.nu}")
            print(f"  - 传感器数量: {self.model.nsensor}")
            
            return True
            
        except Exception as e:
            print(f"✗ MuJoCo初始化失败: {e}")
            return False
    
    def set_initial_pose(self):
        """设置机器人初始姿态"""
        # 设置基座位置 (x, y, z, qw, qx, qy, qz)
        self.data.qpos[0] = 0.0    # x
        self.data.qpos[1] = 0.0    # y  
        self.data.qpos[2] = 1.0    # z - 机器人高度
        self.data.qpos[3] = 1.0    # qw
        self.data.qpos[4] = 0.0    # qx
        self.data.qpos[5] = 0.0    # qy
        self.data.qpos[6] = 0.0    # qz
        
        # 设置关节角度为站立姿态
        joint_angles = {
            'J_hip_l_pitch': -0.3,
            'J_knee_l_pitch': 0.6,
            'J_ankle_l_pitch': -0.3,
            'J_hip_r_pitch': -0.3,
            'J_knee_r_pitch': 0.6,
            'J_ankle_r_pitch': -0.3,
        }
        
        for joint_name, angle in joint_angles.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.data.qpos[joint_id] = angle
    
    def initialize_trajectory_planners(self):
        """初始化足端轨迹规划器"""
        # 获取初始足端位置
        left_foot_pos = self.get_foot_position('left')
        right_foot_pos = self.get_foot_position('right')
        
        # 更新数据总线
        self.data_bus.update('left_foot_position', left_foot_pos)
        self.data_bus.update('right_foot_position', right_foot_pos)
        self.data_bus.update('left_foot_target', left_foot_pos + np.array([0.2, 0.0, 0.1]))
        self.data_bus.update('right_foot_target', right_foot_pos + np.array([0.2, 0.0, 0.1]))
        
        # 创建轨迹配置
        left_config = TrajectoryConfig(
            step_height=0.08,
            swing_duration=1.0,
            interpolation_type="cubic",
            vertical_trajectory_type="sine",
            enable_ground_contact_detection=False  # 测试模式下禁用
        )
        
        right_config = TrajectoryConfig(
            step_height=0.06,
            swing_duration=1.2,
            interpolation_type="cubic", 
            vertical_trajectory_type="sine",
            enable_ground_contact_detection=False  # 测试模式下禁用
        )
        
        # 创建轨迹规划器
        self.left_foot_traj = FootTrajectory(
            foot_name='LF',  # 使用AzureLoong的足部命名
            config=left_config
        )
        
        self.right_foot_traj = FootTrajectory(
            foot_name='RF',  # 使用AzureLoong的足部命名
            config=right_config
        )
        
        # 连接数据总线
        self.left_foot_traj.connect_data_bus(self.data_bus)
        self.right_foot_traj.connect_data_bus(self.data_bus)
        
        print("✓ 足端轨迹规划器初始化完成")
    
    def get_foot_position(self, foot_name):
        """获取足端位置"""
        if foot_name == 'left':
            site_name = 'lf-tc'
        else:
            site_name = 'rf-tc'
            
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def get_touch_sensor_data(self, foot_name):
        """获取触觉传感器数据"""
        if foot_name == 'left':
            sensor_name = 'lf-touch'
        else:
            sensor_name = 'rf-touch'
            
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id >= 0:
            return self.data.sensordata[sensor_id]
        else:
            return 0.0
    
    def update_trajectory_planners(self):
        """更新轨迹规划器"""
        # 获取当前足端位置
        left_pos = self.get_foot_position('left')
        right_pos = self.get_foot_position('right')
        
        # 获取触觉传感器数据
        left_touch = self.get_touch_sensor_data('left')
        right_touch = self.get_touch_sensor_data('right')
        
        # 更新数据总线
        self.data_bus.update('left_foot_position', left_pos)
        self.data_bus.update('right_foot_position', right_pos)
        self.data_bus.update('left_foot_contact_force', left_touch)
        self.data_bus.update('right_foot_contact_force', right_touch)
        
        # 更新轨迹规划器
        if self.left_foot_traj:
            self.left_foot_traj.update(self.dt)
            
        if self.right_foot_traj:
            self.right_foot_traj.update(self.dt)
    
    def generate_predicted_trajectory(self, foot_traj, num_points=50):
        """生成预测轨迹点"""
        if not foot_traj or foot_traj.state.name == 'IDLE':
            return []
            
        predicted_points = []
        current_phase = foot_traj.phase
        
        # 生成未来轨迹点
        for i in range(num_points):
            future_phase = current_phase + (i / num_points) * (1.0 - current_phase)
            if future_phase > 1.0:
                break
                
            # 临时设置相位来计算位置
            original_phase = foot_traj.phase
            foot_traj.phase = future_phase
            
            # 计算该相位下的位置
            pos = foot_traj.current_position
            predicted_points.append(pos.copy())
            
            # 恢复原始相位
            foot_traj.phase = original_phase
            
        return predicted_points
    
    def update_trajectory_history(self):
        """更新轨迹历史记录"""
        left_pos = self.get_foot_position('left')
        right_pos = self.get_foot_position('right')
        
        self.trajectory_history['left_foot'].append(left_pos.copy())
        self.trajectory_history['right_foot'].append(right_pos.copy())
        
        # 更新预测轨迹
        self.predicted_trajectory['left_foot'] = self.generate_predicted_trajectory(self.left_foot_traj)
        self.predicted_trajectory['right_foot'] = self.generate_predicted_trajectory(self.right_foot_traj)
    
    def add_trajectory_visualization(self, viewer):
        """添加轨迹可视化"""
        # 清除之前的可视化对象
        self.clear_trajectory_visualization(viewer)
        
        # 绘制历史轨迹
        self.draw_trajectory_history(viewer)
        
        # 绘制预测轨迹
        self.draw_predicted_trajectory(viewer)
        
        # 绘制目标点
        self.draw_target_points(viewer)
        
        # 绘制当前相位指示器
        self.draw_phase_indicators(viewer)
    
    def draw_trajectory_history(self, viewer):
        """绘制历史轨迹"""
        for foot_name, positions in self.trajectory_history.items():
            if len(positions) < 2:
                continue
                
            color = self.trajectory_colors[foot_name]
            
            # 绘制轨迹线段
            for i in range(len(positions) - 1):
                start_pos = positions[i]
                end_pos = positions[i + 1]
                
                # 添加线段
                viewer.add_marker(
                    pos=start_pos,
                    size=[0.002, 0.002, np.linalg.norm(end_pos - start_pos)],
                    rgba=color,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    label=f"{foot_name}_history_{i}"
                )
    
    def draw_predicted_trajectory(self, viewer):
        """绘制预测轨迹"""
        for foot_name, positions in self.predicted_trajectory.items():
            if len(positions) < 2:
                continue
                
            color = self.trajectory_colors['predicted']
            
            # 绘制预测轨迹点
            for i, pos in enumerate(positions):
                alpha = 1.0 - (i / len(positions)) * 0.5  # 渐变透明度
                point_color = color.copy()
                point_color[3] = alpha
                
                viewer.add_marker(
                    pos=pos,
                    size=[0.008, 0.008, 0.008],
                    rgba=point_color,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=f"{foot_name}_predicted_{i}"
                )
    
    def draw_target_points(self, viewer):
        """绘制目标点"""
        # 左足目标点
        left_target = self.data_bus.get('left_foot_target')
        if left_target is not None:
            viewer.add_marker(
                pos=left_target,
                size=[0.02, 0.02, 0.02],
                rgba=self.trajectory_colors['target'],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="left_target"
            )
        
        # 右足目标点
        right_target = self.data_bus.get('right_foot_target')
        if right_target is not None:
            viewer.add_marker(
                pos=right_target,
                size=[0.02, 0.02, 0.02],
                rgba=self.trajectory_colors['target'],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="right_target"
            )
    
    def draw_phase_indicators(self, viewer):
        """绘制相位进度指示器"""
        # 左足相位指示器
        if self.left_foot_traj and self.left_foot_traj.state.name != 'IDLE':
            left_pos = self.get_foot_position('left')
            phase_height = left_pos[2] + 0.1 + 0.05 * self.left_foot_traj.phase
            
            viewer.add_marker(
                pos=[left_pos[0], left_pos[1], phase_height],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="left_phase"
            )
        
        # 右足相位指示器
        if self.right_foot_traj and self.right_foot_traj.state.name != 'IDLE':
            right_pos = self.get_foot_position('right')
            phase_height = right_pos[2] + 0.1 + 0.05 * self.right_foot_traj.phase
            
            viewer.add_marker(
                pos=[right_pos[0], right_pos[1], phase_height],
                size=[0.01, 0.01, 0.01],
                rgba=[1, 0, 0, 1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="right_phase"
            )
    
    def clear_trajectory_visualization(self, viewer):
        """清除轨迹可视化"""
        # MuJoCo viewer会自动清理标记，无需手动清除
        pass
    
    def print_status(self):
        """打印状态信息"""
        if self.sim_time % 1.0 < self.dt:  # 每秒打印一次
            left_pos = self.get_foot_position('left')
            right_pos = self.get_foot_position('right')
            
            left_phase = self.left_foot_traj.phase if self.left_foot_traj else 0.0
            right_phase = self.right_foot_traj.phase if self.right_foot_traj else 0.0
            
            left_state = self.left_foot_traj.state.name if self.left_foot_traj else 'NONE'
            right_state = self.right_foot_traj.state.name if self.right_foot_traj else 'NONE'
            
            print(f"\n=== 仿真状态 (t={self.sim_time:.1f}s) ===")
            print(f"左足: 位置={left_pos}, 相位={left_phase:.3f}, 状态={left_state}")
            print(f"右足: 位置={right_pos}, 相位={right_phase:.3f}, 状态={right_state}")
    
    def update_targets_periodically(self):
        """周期性更新目标点"""
        # 每5秒更新一次目标点
        if int(self.sim_time) % 5 == 0 and self.sim_time % 1.0 < self.dt:
            # 生成新的随机目标点
            base_pos = np.array([0.0, 0.0, 0.0])
            
            left_target = base_pos + np.array([
                0.3 * np.sin(self.sim_time * 0.5),
                0.1 + 0.15 * np.cos(self.sim_time * 0.3),
                0.0
            ])
            
            right_target = base_pos + np.array([
                0.3 * np.sin(self.sim_time * 0.5 + np.pi),
                -0.1 + 0.15 * np.cos(self.sim_time * 0.3 + np.pi),
                0.0
            ])
            
            # 获取当前足端位置
            left_foot_pos = self.get_foot_position('left')
            right_foot_pos = self.get_foot_position('right')
            
            self.data_bus.update('left_foot_target', left_target)
            self.data_bus.update('right_foot_target', right_target)
            
            # 重启轨迹规划
            if self.left_foot_traj:
                self.left_foot_traj.start_swing(left_foot_pos, left_target)
            if self.right_foot_traj:
                self.right_foot_traj.start_swing(right_foot_pos, right_target)
    
    def run_simulation(self):
        """运行仿真"""
        if not self.initialize_mujoco():
            return
            
        self.initialize_trajectory_planners()
        
        print("\n🚀 启动MuJoCo轨迹可视化演示...")
        print("📋 控制说明:")
        print("  - 鼠标拖拽: 旋转视角")
        print("  - 鼠标滚轮: 缩放")
        print("  - 右键拖拽: 平移")
        print("  - ESC: 退出")
        print("\n🎨 可视化说明:")
        print("  - 绿色轨迹: 左足历史轨迹")
        print("  - 红色轨迹: 右足历史轨迹") 
        print("  - 蓝色点: 预测轨迹")
        print("  - 黄色球: 目标位置")
        print("  - 绿/红小球: 相位进度指示器")
        
        # 启动轨迹规划
        left_start = self.get_foot_position('left')
        right_start = self.get_foot_position('right')
        left_target = self.data_bus.get('left_foot_target')
        right_target = self.data_bus.get('right_foot_target')
        
        if self.left_foot_traj and left_target is not None:
            self.left_foot_traj.start_swing(left_start, left_target)
        if self.right_foot_traj and right_target is not None:
            self.right_foot_traj.start_swing(right_start, right_target)
        
        # 尝试启动可视化
        try:
            self._run_with_viewer()
        except RuntimeError as e:
            if "mjpython" in str(e):
                print(f"\n⚠️  MuJoCo可视化需要mjpython运行环境")
                print("💡 解决方案:")
                print("   1. 安装MuJoCo: pip install mujoco")
                print("   2. 使用mjpython运行: mjpython run_trajectory_visualization.py")
                print("   3. 或者运行无GUI模式的轨迹演示")
                print("\n🔄 切换到无GUI模式...")
                self._run_without_viewer()
            else:
                raise e
    
    def _run_with_viewer(self):
        """使用MuJoCo viewer运行仿真"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                
                # 更新仿真时间
                self.sim_time = time.time() - start_time
                
                # 更新轨迹规划器
                self.update_trajectory_planners()
                
                # 更新轨迹历史
                self.update_trajectory_history()
                
                # 周期性更新目标点
                self.update_targets_periodically()
                
                # 前向运动学
                mujoco.mj_step(self.model, self.data)
                
                # 添加轨迹可视化
                self.add_trajectory_visualization(viewer)
                
                # 同步viewer
                viewer.sync()
                
                # 打印状态
                self.print_status()
                
                # 控制帧率
                elapsed = time.time() - step_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
    
    def _run_without_viewer(self):
        """无GUI模式运行仿真，输出轨迹数据"""
        print("\n📊 无GUI模式轨迹演示")
        print("=" * 50)
        
        start_time = time.time()
        max_sim_time = 10.0  # 运行10秒
        
        # 创建轨迹数据记录
        trajectory_log = {
            'time': [],
            'left_foot_pos': [],
            'right_foot_pos': [],
            'left_foot_target': [],
            'right_foot_target': [],
            'left_phase': [],
            'right_phase': [],
            'left_state': [],
            'right_state': []
        }
        
        while self.sim_time < max_sim_time:
            step_start = time.time()
            
            # 更新仿真时间
            self.sim_time = time.time() - start_time
            
            # 更新轨迹规划器
            self.update_trajectory_planners()
            
            # 更新轨迹历史
            self.update_trajectory_history()
            
            # 周期性更新目标点
            self.update_targets_periodically()
            
            # 前向运动学
            mujoco.mj_step(self.model, self.data)
            
            # 记录轨迹数据
            if len(trajectory_log['time']) == 0 or self.sim_time - trajectory_log['time'][-1] > 0.1:
                left_pos = self.get_foot_position('left')
                right_pos = self.get_foot_position('right')
                left_target = self.data_bus.get('left_foot_target')
                right_target = self.data_bus.get('right_foot_target')
                
                trajectory_log['time'].append(self.sim_time)
                trajectory_log['left_foot_pos'].append(left_pos.copy())
                trajectory_log['right_foot_pos'].append(right_pos.copy())
                trajectory_log['left_foot_target'].append(left_target.copy() if left_target is not None else np.zeros(3))
                trajectory_log['right_foot_target'].append(right_target.copy() if right_target is not None else np.zeros(3))
                trajectory_log['left_phase'].append(self.left_foot_traj.phase if self.left_foot_traj else 0.0)
                trajectory_log['right_phase'].append(self.right_foot_traj.phase if self.right_foot_traj else 0.0)
                trajectory_log['left_state'].append(self.left_foot_traj.state.name if self.left_foot_traj else 'NONE')
                trajectory_log['right_state'].append(self.right_foot_traj.state.name if self.right_foot_traj else 'NONE')
            
            # 打印状态
            self.print_status()
            
            # 控制帧率
            elapsed = time.time() - step_start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
        
        # 输出轨迹统计信息
        self._print_trajectory_summary(trajectory_log)
        
        # 保存轨迹数据
        self._save_trajectory_data(trajectory_log)
    
    def _print_trajectory_summary(self, trajectory_log):
        """打印轨迹摘要信息"""
        print("\n📈 轨迹摘要统计")
        print("=" * 50)
        
        if len(trajectory_log['time']) > 0:
            total_time = trajectory_log['time'][-1]
            left_positions = np.array(trajectory_log['left_foot_pos'])
            right_positions = np.array(trajectory_log['right_foot_pos'])
            
            # 计算轨迹长度
            left_distances = np.linalg.norm(np.diff(left_positions, axis=0), axis=1)
            right_distances = np.linalg.norm(np.diff(right_positions, axis=0), axis=1)
            
            left_total_distance = np.sum(left_distances)
            right_total_distance = np.sum(right_distances)
            
            # 计算最大高度
            left_max_height = np.max(left_positions[:, 2])
            right_max_height = np.max(right_positions[:, 2])
            
            print(f"总仿真时间: {total_time:.2f}s")
            print(f"数据点数量: {len(trajectory_log['time'])}")
            print(f"\n左足轨迹:")
            print(f"  - 总移动距离: {left_total_distance:.3f}m")
            print(f"  - 最大高度: {left_max_height:.3f}m")
            print(f"  - 最终位置: {left_positions[-1]}")
            print(f"\n右足轨迹:")
            print(f"  - 总移动距离: {right_total_distance:.3f}m")
            print(f"  - 最大高度: {right_max_height:.3f}m")
            print(f"  - 最终位置: {right_positions[-1]}")
            
            # 相位统计
            left_phases = trajectory_log['left_phase']
            right_phases = trajectory_log['right_phase']
            print(f"\n相位统计:")
            print(f"  - 左足最大相位: {max(left_phases):.3f}")
            print(f"  - 右足最大相位: {max(right_phases):.3f}")
    
    def _save_trajectory_data(self, trajectory_log):
        """保存轨迹数据到文件"""
        try:
            import json
            
            # 转换numpy数组为列表以便JSON序列化
            json_data = {}
            for key, values in trajectory_log.items():
                if key in ['left_foot_pos', 'right_foot_pos', 'left_foot_target', 'right_foot_target']:
                    json_data[key] = [pos.tolist() for pos in values]
                else:
                    json_data[key] = values
            
            filename = f"trajectory_data_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"\n💾 轨迹数据已保存到: {filename}")
            
        except Exception as e:
            print(f"\n⚠️  保存轨迹数据失败: {e}")

def main():
    """主函数"""
    print("🤖 AzureLoong机器人轨迹可视化演示")
    print("=" * 50)
    
    # 检查模型文件
    model_path = "models/scene.xml"
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        print("请确保在项目根目录运行此脚本")
        return
    
    # 创建并运行可视化器
    visualizer = MuJoCoTrajectoryVisualizer(model_path)
    
    try:
        visualizer.run_simulation()
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 