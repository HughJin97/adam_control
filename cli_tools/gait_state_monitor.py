#!/usr/bin/env python3
"""
步态状态监控脚本

使用matplotlib实时显示：
1. 步态状态时间序列
2. 足部目标位置轨迹
3. 足底力变化图表

作者: Adam Control Team
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple
import sys
import warnings

# 忽略字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 导入步态相关模块
try:
    from data_bus import DataBus
    from foot_placement import FootPlacementPlanner, FootPlacementConfig, Vector3D
    from gait_scheduler import GaitScheduler, GaitState, GaitSchedulerConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class GaitStateMonitor:
    """步态状态监控类"""
    
    def __init__(self, history_length: int = 1000):
        """初始化监控器"""
        self.history_length = history_length
        
        # 初始化步态系统
        self.data_bus = DataBus()
        self.foot_planner = FootPlacementPlanner(FootPlacementConfig())
        
        gait_config = GaitSchedulerConfig()
        gait_config.touchdown_force_threshold = 30.0
        self.gait_scheduler = GaitScheduler(gait_config)
        
        # 数据存储
        self.time_history = deque(maxlen=history_length)
        self.state_history = deque(maxlen=history_length)
        self.left_force_history = deque(maxlen=history_length)
        self.right_force_history = deque(maxlen=history_length)
        self.left_target_history = deque(maxlen=history_length)
        self.right_target_history = deque(maxlen=history_length)
        
        # 状态映射
        self.state_colors = {
            GaitState.IDLE: 0,
            GaitState.STANDING: 1,
            GaitState.LEFT_SUPPORT: 2,
            GaitState.RIGHT_SUPPORT: 3,
            GaitState.DOUBLE_SUPPORT_LR: 4,
            GaitState.DOUBLE_SUPPORT_RL: 5,
        }
        
        self.state_names = {
            0: 'IDLE',
            1: 'STANDING', 
            2: 'LEFT_SUPPORT',
            3: 'RIGHT_SUPPORT',
            4: 'DOUBLE_LR',
            5: 'DOUBLE_RL',
        }
        
        self.state_color_map = {
            0: 'gray',
            1: 'lightgray',
            2: 'green',
            3: 'blue', 
            4: 'yellow',
            5: 'orange',
        }
        
        # 仿真参数
        self.dt = 0.001  # 1ms
        self.start_time = time.time()
        self.current_time = 0.0
        
        # 仿真足部位置（简化）
        self.left_foot_pos = [0.0, 0.1, 0.0]
        self.right_foot_pos = [0.0, -0.1, 0.0]
        
        print("步态状态监控器初始化完成")
    
    def update_gait_system(self):
        """更新步态系统"""
        # 简化的接触力仿真
        left_force = 50.0 if abs(self.left_foot_pos[2]) < 0.02 else 0.0
        right_force = 50.0 if abs(self.right_foot_pos[2]) < 0.02 else 0.0
        
        # 模拟步行过程中的足部抬起
        if self.gait_scheduler.current_state == GaitState.LEFT_SUPPORT:
            # 右腿摆动，抬起右脚
            self.right_foot_pos[2] = 0.05 * np.sin(self.current_time * 5)
            right_force = 0.0
        elif self.gait_scheduler.current_state == GaitState.RIGHT_SUPPORT:
            # 左腿摆动，抬起左脚
            self.left_foot_pos[2] = 0.05 * np.sin(self.current_time * 5)
            left_force = 0.0
        else:
            # 双支撑或站立，两脚着地
            self.left_foot_pos[2] = 0.0
            self.right_foot_pos[2] = 0.0
        
        # 更新数据总线
        self.data_bus.set_end_effector_contact_force("left_foot", left_force)
        self.data_bus.set_end_effector_contact_force("right_foot", right_force)
        
        # 更新步态调度器
        left_vel = np.zeros(3)
        right_vel = np.zeros(3)
        self.gait_scheduler.update_sensor_data(left_force, right_force, left_vel, right_vel)
        self.gait_scheduler.update_gait_state(self.dt)
        
        # 足步规划
        self._update_foot_planning()
        
        # 记录数据
        self.time_history.append(self.current_time)
        self.state_history.append(self.state_colors.get(self.gait_scheduler.current_state, 0))
        self.left_force_history.append(left_force)
        self.right_force_history.append(right_force)
        
        self.current_time += self.dt
    
    def _update_foot_planning(self):
        """更新足步规划"""
        current_state = self.gait_scheduler.current_state
        
        # 根据当前步态状态触发足步规划
        if current_state == GaitState.LEFT_SUPPORT:  # 右腿摆动
            swing_leg = "right"
            support_leg = "left"
        elif current_state == GaitState.RIGHT_SUPPORT:  # 左腿摆动
            swing_leg = "left"
            support_leg = "right" 
        else:
            # 使用当前位置作为目标
            self.left_target_history.append((self.left_foot_pos[0], self.left_foot_pos[1]))
            self.right_target_history.append((self.right_foot_pos[0], self.right_foot_pos[1]))
            return
        
        # 设置运动意图（前进）
        self.foot_planner.set_body_motion_intent(Vector3D(0.1, 0.0, 0.0), 0.0)
        
        # 执行足步规划
        try:
            target_position = self.foot_planner.plan_foot_placement(swing_leg, support_leg)
            
            if swing_leg == "left":
                self.left_target_history.append((target_position.x, target_position.y))
                self.right_target_history.append((self.right_foot_pos[0], self.right_foot_pos[1]))
            else:
                self.left_target_history.append((self.left_foot_pos[0], self.left_foot_pos[1]))
                self.right_target_history.append((target_position.x, target_position.y))
            
        except Exception as e:
            print(f"足步规划失败: {e}")
            # 使用当前位置
            self.left_target_history.append((self.left_foot_pos[0], self.left_foot_pos[1]))
            self.right_target_history.append((self.right_foot_pos[0], self.right_foot_pos[1]))


class GaitVisualizationPlotter:
    """步态可视化绘图类"""
    
    def __init__(self, monitor: GaitStateMonitor):
        """初始化绘图器"""
        self.monitor = monitor
        
        # 配置matplotlib以支持中文（使用英文避免字体问题）
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形和子图
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Gait State Real-time Monitor', fontsize=16, fontweight='bold')
        
        # 子图1: 步态状态时间序列
        self.ax_state = self.axes[0, 0]
        self.ax_state.set_title('Gait State Changes')
        self.ax_state.set_xlabel('Time (s)')
        self.ax_state.set_ylabel('Gait State')
        self.ax_state.set_ylim(-0.5, 5.5)
        self.ax_state.grid(True, alpha=0.3)
        
        # 设置状态标签
        state_labels = ['IDLE', 'STANDING', 'LEFT_SUP', 'RIGHT_SUP', 'DOUBLE_LR', 'DOUBLE_RL']
        self.ax_state.set_yticks(range(6))
        self.ax_state.set_yticklabels(state_labels)
        
        # 子图2: 足底力
        self.ax_force = self.axes[0, 1] 
        self.ax_force.set_title('Foot Force Changes')
        self.ax_force.set_xlabel('Time (s)')
        self.ax_force.set_ylabel('Force (N)')
        self.ax_force.grid(True, alpha=0.3)
        
        # 子图3: 足部位置轨迹
        self.ax_trajectory = self.axes[1, 0]
        self.ax_trajectory.set_title('Foot Target Position Trajectory')
        self.ax_trajectory.set_xlabel('X Position (m)')
        self.ax_trajectory.set_ylabel('Y Position (m)')
        self.ax_trajectory.set_aspect('equal')
        self.ax_trajectory.grid(True, alpha=0.3)
        
        # 子图4: 当前状态显示
        self.ax_status = self.axes[1, 1]
        self.ax_status.set_title('Current Status')
        self.ax_status.axis('off')
        
        # 初始化绘图元素
        self.state_line, = self.ax_state.plot([], [], 'b-', linewidth=2, label='Gait State')
        self.left_force_line, = self.ax_force.plot([], [], 'r-', linewidth=2, label='Left Foot')
        self.right_force_line, = self.ax_force.plot([], [], 'b-', linewidth=2, label='Right Foot')
        self.left_traj_line, = self.ax_trajectory.plot([], [], 'ro-', markersize=4, alpha=0.7, label='Left Target')
        self.right_traj_line, = self.ax_trajectory.plot([], [], 'bo-', markersize=4, alpha=0.7, label='Right Target')
        
        # 当前位置标记
        self.left_current = Circle((0, 0.1), 0.02, color='red', alpha=0.8)
        self.right_current = Circle((0, -0.1), 0.02, color='blue', alpha=0.8)
        self.ax_trajectory.add_patch(self.left_current)
        self.ax_trajectory.add_patch(self.right_current)
        
        # 图例
        self.ax_force.legend()
        self.ax_trajectory.legend()
        
        # 状态文本
        self.status_text = self.ax_status.text(0.1, 0.9, '', transform=self.ax_status.transAxes,
                                              fontsize=12, verticalalignment='top', family='monospace')
        
        # 开始步行
        self.monitor.gait_scheduler.start_walking()
    
    def animate(self, frame):
        """动画更新函数"""
        # 更新步态系统
        self.monitor.update_gait_system()
        
        if not self.monitor.time_history:
            return self.state_line, self.left_force_line, self.right_force_line, \
                   self.left_traj_line, self.right_traj_line
        
        # 获取数据
        times = list(self.monitor.time_history)
        states = list(self.monitor.state_history)
        left_forces = list(self.monitor.left_force_history)
        right_forces = list(self.monitor.right_force_history)
        
        # 更新步态状态图
        self.state_line.set_data(times, states)
        
        # 为不同状态添加颜色
        for i, state in enumerate(states):
            if i > 0 and i < len(times):
                color = self.monitor.state_color_map.get(state, 'gray')
                self.ax_state.axhspan(state-0.1, state+0.1, 
                                     xmin=(times[i-1]-times[0])/(times[-1]-times[0]) if times[-1] > times[0] else 0,
                                     xmax=(times[i]-times[0])/(times[-1]-times[0]) if times[-1] > times[0] else 1,
                                     alpha=0.3, color=color)
        
        # 更新坐标轴范围
        if times:
            self.ax_state.set_xlim(times[0], max(times[-1], times[0] + 1))
            self.ax_force.set_xlim(times[0], max(times[-1], times[0] + 1))
        
        # 更新足底力图
        self.left_force_line.set_data(times, left_forces)
        self.right_force_line.set_data(times, right_forces)
        
        if left_forces or right_forces:
            all_forces = left_forces + right_forces
            max_force = max(all_forces) if all_forces else 100
            self.ax_force.set_ylim(0, max_force * 1.1)
        
        # 更新足部轨迹
        if self.monitor.left_target_history and self.monitor.right_target_history:
            left_targets = list(self.monitor.left_target_history)
            right_targets = list(self.monitor.right_target_history)
            
            if left_targets:
                left_x, left_y = zip(*left_targets[-50:])  # 显示最近50个点
                self.left_traj_line.set_data(left_x, left_y)
                
                # 更新当前位置
                self.left_current.center = (left_x[-1], left_y[-1])
            
            if right_targets:
                right_x, right_y = zip(*right_targets[-50:])  # 显示最近50个点
                self.right_traj_line.set_data(right_x, right_y)
                
                # 更新当前位置
                self.right_current.center = (right_x[-1], right_y[-1])
            
            # 自动调整轨迹图范围
            all_x = list(left_x) + list(right_x) if left_targets and right_targets else []
            all_y = list(left_y) + list(right_y) if left_targets and right_targets else []
            
            if all_x and all_y:
                margin = 0.1
                self.ax_trajectory.set_xlim(min(all_x) - margin, max(all_x) + margin)
                self.ax_trajectory.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # 更新状态文本（使用英文）
        current_state = self.monitor.gait_scheduler.current_state
        left_force = self.monitor.left_force_history[-1] if self.monitor.left_force_history else 0
        right_force = self.monitor.right_force_history[-1] if self.monitor.right_force_history else 0
        
        status_info = f"""Current State: {current_state.value}
Time: {self.monitor.current_time:.2f}s

Foot Force:
  Left:  {left_force:.1f}N
  Right: {right_force:.1f}N

Swing Leg: {getattr(self.monitor.gait_scheduler, 'swing_leg', 'none')}
Support Leg: {getattr(self.monitor.gait_scheduler, 'support_leg', 'both')}

Control:
  Ctrl+C: Exit Monitor
"""
        
        self.status_text.set_text(status_info)
        
        return self.state_line, self.left_force_line, self.right_force_line, \
               self.left_traj_line, self.right_traj_line
    
    def start_monitoring(self, interval: int = 50):
        """开始监控"""
        print("Starting gait state real-time monitoring...")
        print("Press Ctrl+C to exit")
        
        # 创建动画
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, blit=False, cache_frame_data=False
        )
        
        # 显示图形
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gait State Monitor")
    parser.add_argument("--history", type=int, default=1000,
                       help="History data length")
    parser.add_argument("--interval", type=int, default=50,
                       help="Update interval (ms)")
    
    args = parser.parse_args()
    
    try:
        # 创建监控器
        monitor = GaitStateMonitor(args.history)
        
        # 创建可视化器
        plotter = GaitVisualizationPlotter(monitor)
        
        # 开始监控
        plotter.start_monitoring(args.interval)
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 