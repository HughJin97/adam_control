#!/usr/bin/env python3
"""
步态可视化仿真脚本

功能特性:
1. 实时显示当前legState (文字/颜色标注)
2. 可视化target_foot_pos (小球/透明标记)
3. 集成完整的步态调度和足步规划系统
4. 交互式控制界面

作者: Adam Control Team
"""

import mujoco as mj
import numpy as np
import time
import threading
from typing import Dict, Optional, Tuple, List
import sys
import os

# 导入步态相关模块
try:
    from data_bus import DataBus
    from foot_placement import FootPlacementPlanner, FootPlacementConfig, Vector3D
    from gait_scheduler import GaitScheduler, GaitState, GaitSchedulerConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)


class GaitVisualizationSimulation:
    """步态可视化仿真类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化仿真"""
        self.model_path = model_path
        
        # 尝试加载模型
        try:
            self.model = mj.MjModel.from_xml_path(model_path)
            self.data = mj.MjData(self.model)
        except Exception as e:
            print(f"无法加载模型文件 {model_path}: {e}")
            print("使用简单的双足机器人模型...")
            self._create_simple_biped_model()
        
        # 初始化步态系统
        self.data_bus = DataBus()
        self.foot_planner = FootPlacementPlanner(FootPlacementConfig())
        
        gait_config = GaitSchedulerConfig()
        gait_config.touchdown_force_threshold = 30.0
        gait_config.swing_duration = 0.4  # 400ms摆动时间
        gait_config.double_support_duration = 0.1  # 100ms双支撑时间
        self.gait_scheduler = GaitScheduler(gait_config)
        
        # 可视化参数
        self.target_foot_markers = {}  # 足部目标位置标记
        self.leg_state_colors = {
            GaitState.LEFT_SUPPORT: [0.0, 1.0, 0.0, 0.8],    # 绿色 - 左腿支撑
            GaitState.RIGHT_SUPPORT: [0.0, 0.0, 1.0, 0.8],   # 蓝色 - 右腿支撑  
            GaitState.DOUBLE_SUPPORT_LR: [1.0, 1.0, 0.0, 0.8],  # 黄色 - 双支撑LR
            GaitState.DOUBLE_SUPPORT_RL: [1.0, 0.5, 0.0, 0.8],  # 橙色 - 双支撑RL
            GaitState.STANDING: [0.5, 0.5, 0.5, 0.8],        # 灰色 - 站立
            GaitState.IDLE: [0.3, 0.3, 0.3, 0.8],            # 深灰 - 空闲
        }
        
        # 仿真控制
        self.running = False
        self.paused = False
        self.simulation_speed = 1.0
        self.control_frequency = 1000.0  # 1kHz控制频率
        self.dt = 1.0 / self.control_frequency
        
        # 步态控制参数
        self.step_length = 0.1  # 步长10cm
        self.step_height = 0.05  # 抬脚高度5cm
        self.step_duration = 0.5  # 单步时长500ms
        
        # 目标位置和轨迹
        self.left_foot_target = None
        self.right_foot_target = None
        self.swing_start_time = {}
        self.swing_start_pos = {}
        
        # 性能监控
        self.gait_cycle_count = 0
        self.last_state_change_time = time.time()
        self.state_history = []
        self.foot_contact_history = []
        
        # 渲染相关
        self.renderer = None
        self.viewer = None
        self.render_context = None
        
        print("步态可视化仿真初始化完成")
    
    def _create_simple_biped_model(self):
        """创建简单的双足机器人模型"""
        xml_string = """
        <mujoco model="simple_biped">
          <compiler angle="degree"/>
          
          <default>
            <joint damping="0.5"/>
            <geom material="robot" contype="1" conaffinity="1"/>
          </default>
          
          <asset>
            <material name="robot" rgba="0.7 0.7 0.9 1"/>
            <material name="ground" rgba="0.2 0.8 0.2 1"/>
            <material name="target" rgba="1.0 0.2 0.2 0.5"/>
            <material name="support_leg" rgba="0.2 1.0 0.2 1"/>
            <material name="swing_leg" rgba="1.0 0.2 0.2 1"/>
          </asset>
          
          <worldbody>
            <!-- 地面 -->
            <geom name="ground" type="plane" size="5 5 0.1" material="ground" contype="1" conaffinity="1"/>
            
            <!-- 躯干 -->
            <body name="torso" pos="0 0 0.9">
              <geom name="torso" type="box" size="0.15 0.08 0.25" rgba="0.7 0.7 0.9 1"/>
              <joint name="root_z" type="slide" axis="0 0 1"/>
              <joint name="root_x" type="slide" axis="1 0 0"/>
              <joint name="root_y" type="slide" axis="0 1 0"/>
              
              <!-- 左腿 -->
              <body name="left_thigh" pos="0 0.1 -0.25">
                <joint name="left_hip" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="left_thigh" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.5 0.5 0.9 1"/>
                
                <body name="left_shin" pos="0 0 -0.3">
                  <joint name="left_knee" type="hinge" axis="0 1 0" range="-120 0"/>
                  <geom name="left_shin" type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.3 0.3 0.7 1"/>
                  
                  <body name="left_foot" pos="0 0 -0.3">
                    <joint name="left_ankle" type="hinge" axis="0 1 0" range="-30 30"/>
                    <geom name="left_foot" type="box" size="0.08 0.04 0.02" rgba="0.2 0.2 0.5 1" contype="1" conaffinity="1"/>
                    <!-- 足底传感器站点 -->
                    <site name="left_foot_sensor" pos="0 0 -0.02" size="0.01"/>
                  </body>
                </body>
              </body>
              
              <!-- 右腿 -->
              <body name="right_thigh" pos="0 -0.1 -0.25">
                <joint name="right_hip" type="hinge" axis="0 1 0" range="-90 90"/>
                <geom name="right_thigh" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.5 0.5 0.9 1"/>
                
                <body name="right_shin" pos="0 0 -0.3">
                  <joint name="right_knee" type="hinge" axis="0 1 0" range="-120 0"/>
                  <geom name="right_shin" type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.3 0.3 0.7 1"/>
                  
                  <body name="right_foot" pos="0 0 -0.3">
                    <joint name="right_ankle" type="hinge" axis="0 1 0" range="-30 30"/>
                    <geom name="right_foot" type="box" size="0.08 0.04 0.02" rgba="0.2 0.2 0.5 1" contype="1" conaffinity="1"/>
                    <!-- 足底传感器站点 -->
                    <site name="right_foot_sensor" pos="0 0 -0.02" size="0.01"/>
                  </body>
                </body>
              </body>
            </body>
            
            <!-- 可视化目标位置标记 -->
            <body name="left_target" pos="0 0.1 0.05">
              <geom name="left_target_marker" type="sphere" size="0.03" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
            </body>
            
            <body name="right_target" pos="0 -0.1 0.05">
              <geom name="right_target_marker" type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
          </worldbody>
          
          <actuator>
            <motor name="left_hip_motor" joint="left_hip" gear="200" ctrllimited="true" ctrlrange="-100 100"/>
            <motor name="left_knee_motor" joint="left_knee" gear="200" ctrllimited="true" ctrlrange="-100 100"/>
            <motor name="left_ankle_motor" joint="left_ankle" gear="100" ctrllimited="true" ctrlrange="-50 50"/>
            <motor name="right_hip_motor" joint="right_hip" gear="200" ctrllimited="true" ctrlrange="-100 100"/>
            <motor name="right_knee_motor" joint="right_knee" gear="200" ctrllimited="true" ctrlrange="-100 100"/>
            <motor name="right_ankle_motor" joint="right_ankle" gear="100" ctrllimited="true" ctrlrange="-50 50"/>
          </actuator>
          
          <sensor>
            <!-- 足底力传感器 -->
            <force name="left_foot_force" site="left_foot_sensor"/>
            <force name="right_foot_force" site="right_foot_sensor"/>
          </sensor>
        </mujoco>
        """
        
        self.model = mj.MjModel.from_xml_string(xml_string)
        self.data = mj.MjData(self.model)
        print("使用内置简单双足机器人模型")
    
    def update_gait_system(self):
        """更新步态系统"""
        # 读取足底力传感器
        left_force = self._get_foot_force("left")
        right_force = self._get_foot_force("right")
        
        # 更新数据总线
        self.data_bus.set_end_effector_contact_force("left_foot", left_force)
        self.data_bus.set_end_effector_contact_force("right_foot", right_force)
        
        # 获取足部速度
        left_vel = self._get_foot_velocity("left")
        right_vel = self._get_foot_velocity("right")
        
        # 更新步态调度器传感器数据
        self.gait_scheduler.update_sensor_data(left_force, right_force, left_vel, right_vel)
        
        # 记录状态变化前的状态
        prev_state = self.gait_scheduler.current_state
        
        # 更新步态状态
        self.gait_scheduler.update_gait_state(self.dt)
        
        # 检测状态变化
        if prev_state != self.gait_scheduler.current_state:
            current_time = time.time()
            duration = current_time - self.last_state_change_time
            self.last_state_change_time = current_time
            
            print(f"State transition: {prev_state.value} -> {self.gait_scheduler.current_state.value} (duration: {duration:.3f}s)")
            
            # 记录状态历史
            self.state_history.append({
                'time': current_time,
                'prev_state': prev_state,
                'new_state': self.gait_scheduler.current_state,
                'duration': duration
            })
            
            # 如果完成一个完整周期
            if prev_state == GaitState.RIGHT_SUPPORT and self.gait_scheduler.current_state == GaitState.DOUBLE_SUPPORT_RL:
                self.gait_cycle_count += 1
                print(f"Completed gait cycle #{self.gait_cycle_count}")
        
        # 记录接触状态
        self.foot_contact_history.append({
            'time': time.time(),
            'left_contact': left_force > 10.0,
            'right_contact': right_force > 10.0,
            'state': self.gait_scheduler.current_state
        })
        
        # 触发足步规划（在进入摆动相时）
        if prev_state != self.gait_scheduler.current_state:
            self._update_foot_planning()
        
        # 更新可视化
        self._update_visualization()
    
    def _get_foot_force(self, side: str) -> float:
        """获取足底力"""
        if hasattr(self.model, 'sensor_name2id'):
            try:
                sensor_id = self.model.sensor_name2id(f"{side}_foot_force")
                # 力传感器返回3个分量，取z方向的力
                force_z = -self.data.sensordata[sensor_id * 3 + 2]  # 取负值因为z轴向下
                return max(0, force_z)
            except:
                pass
        
        # 备用方案：基于接触检测
        return self._estimate_foot_force_from_contact(side)
    
    def _estimate_foot_force_from_contact(self, side: str) -> float:
        """基于接触检测估算足底力"""
        foot_geom_name = f"{side}_foot"
        try:
            foot_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, foot_geom_name)
            
            # 检查所有接触对
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # 检查是否涉及足部
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # 简化：基于穿透深度估算力
                    penetration = contact.dist
                    if penetration < 0:  # 有穿透
                        force = -penetration * 10000  # 简单的弹簧模型
                        return min(force, 200.0)  # 限制最大力
            
            return 0.0
        except:
            return 0.0
    
    def _get_foot_velocity(self, side: str) -> np.ndarray:
        """获取足部速度"""
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_foot")
            if body_id >= 0:
                # 获取body的速度
                vel = np.zeros(6)
                mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body_id, vel, 0)
                return vel[3:6]  # 返回线速度部分
        except:
            pass
        return np.zeros(3)
    
    def _update_foot_planning(self):
        """更新足步规划"""
        current_state = self.gait_scheduler.current_state
        
        # 根据当前步态状态触发足步规划
        if current_state == GaitState.LEFT_SUPPORT:  # 右腿摆动
            swing_leg = "right"
            support_leg = "left"
            self._plan_swing_trajectory(swing_leg, support_leg)
            
        elif current_state == GaitState.RIGHT_SUPPORT:  # 左腿摆动
            swing_leg = "left"
            support_leg = "right"
            self._plan_swing_trajectory(swing_leg, support_leg)
    
    def _plan_swing_trajectory(self, swing_leg: str, support_leg: str):
        """规划摆动腿轨迹"""
        # 获取当前足部位置
        swing_foot_pos = self._get_foot_position(swing_leg)
        support_foot_pos = self._get_foot_position(support_leg)
        
        # 记录摆动开始时间和位置
        self.swing_start_time[swing_leg] = time.time()
        self.swing_start_pos[swing_leg] = swing_foot_pos.copy()
        
        # 设置运动意图（前进）
        self.foot_planner.set_body_motion_intent(Vector3D(self.step_length, 0.0, 0.0), 0.0)
        
        # 执行足步规划
        try:
            target_position = self.foot_planner.plan_foot_placement(swing_leg, support_leg)
            
            # 存储目标位置用于可视化和控制
            if swing_leg == "left":
                self.left_foot_target = np.array([target_position.x, target_position.y, target_position.z])
            else:
                self.right_foot_target = np.array([target_position.x, target_position.y, target_position.z])
            
            # 更新可视化标记
            foot_key = f"{swing_leg}_foot"
            self.target_foot_markers[foot_key] = {
                'position': target_position,
                'active': True
            }
            
            print(f"Planned {swing_leg} foot target: ({target_position.x:.3f}, {target_position.y:.3f}, {target_position.z:.3f})")
            
        except Exception as e:
            print(f"足步规划失败: {e}")
    
    def _get_foot_position(self, side: str) -> np.ndarray:
        """获取足部位置"""
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_foot")
            if body_id >= 0:
                return self.data.xpos[body_id].copy()
        except:
            pass
        return np.array([0.0, 0.1 if side == "left" else -0.1, 0.0])
    
    def _compute_swing_trajectory(self, swing_leg: str, phase: float) -> np.ndarray:
        """计算摆动腿轨迹"""
        if swing_leg not in self.swing_start_pos:
            return self._get_foot_position(swing_leg)
        
        start_pos = self.swing_start_pos[swing_leg]
        
        # 获取目标位置
        if swing_leg == "left" and self.left_foot_target is not None:
            target_pos = self.left_foot_target
        elif swing_leg == "right" and self.right_foot_target is not None:
            target_pos = self.right_foot_target
        else:
            target_pos = start_pos + np.array([self.step_length, 0, 0])
        
        # 生成平滑轨迹
        # X和Y方向：线性插值
        xy_pos = start_pos[:2] + phase * (target_pos[:2] - start_pos[:2])
        
        # Z方向：抛物线轨迹
        if phase < 0.5:
            # 上升阶段
            z_phase = 2 * phase
            z_pos = start_pos[2] + self.step_height * (4 * z_phase * (1 - z_phase))
        else:
            # 下降阶段
            z_phase = 2 * (phase - 0.5)
            z_pos = start_pos[2] + self.step_height * (1 - z_phase) ** 2
        
        return np.array([xy_pos[0], xy_pos[1], z_pos])
    
    def _walking_control(self):
        """步行控制"""
        current_state = self.gait_scheduler.current_state
        
        # 基础姿态参数
        hip_angle = 0.0
        knee_angle = 0.0
        ankle_angle = 0.0
        
        # 根据状态执行不同的控制策略
        if current_state == GaitState.STANDING or current_state == GaitState.IDLE:
            # 站立姿态
            self._apply_standing_pose()
            
        elif current_state == GaitState.LEFT_SUPPORT:
            # 左腿支撑，右腿摆动
            self._control_support_leg("left")
            self._control_swing_leg("right")
            
        elif current_state == GaitState.RIGHT_SUPPORT:
            # 右腿支撑，左腿摆动
            self._control_support_leg("right")
            self._control_swing_leg("left")
            
        elif current_state in [GaitState.DOUBLE_SUPPORT_LR, GaitState.DOUBLE_SUPPORT_RL]:
            # 双支撑阶段
            self._control_double_support()
    
    def _apply_standing_pose(self):
        """应用站立姿态"""
        # 目标关节角度（直立姿态）
        target_angles = {
            'left_hip': -5.0,
            'left_knee': -10.0,
            'left_ankle': 5.0,
            'right_hip': -5.0,
            'right_knee': -10.0,
            'right_ankle': 5.0,
        }
        
        # 应用PD控制
        for joint_name, target_angle in target_angles.items():
            self._apply_joint_pd_control(joint_name, np.radians(target_angle))
    
    def _control_support_leg(self, side: str):
        """控制支撑腿"""
        # 支撑腿需要保持稳定并承重
        hip_angle = -10.0  # 轻微前倾
        knee_angle = -15.0  # 轻微弯曲
        ankle_angle = 5.0   # 适应地面
        
        self._apply_joint_pd_control(f"{side}_hip", np.radians(hip_angle))
        self._apply_joint_pd_control(f"{side}_knee", np.radians(knee_angle))
        self._apply_joint_pd_control(f"{side}_ankle", np.radians(ankle_angle))
    
    def _control_swing_leg(self, side: str):
        """控制摆动腿"""
        # 计算摆动相位
        if side in self.swing_start_time:
            elapsed = time.time() - self.swing_start_time[side]
            phase = min(elapsed / self.step_duration, 1.0)
        else:
            phase = 0.0
        
        # 根据相位计算期望足部位置
        desired_foot_pos = self._compute_swing_trajectory(side, phase)
        
        # 使用逆运动学或简化控制
        # 这里使用简化的控制策略
        if phase < 0.3:
            # 抬腿阶段
            hip_angle = 30.0 * phase / 0.3
            knee_angle = -60.0 * phase / 0.3
            ankle_angle = 20.0 * phase / 0.3
        elif phase < 0.7:
            # 摆动阶段
            hip_angle = 30.0 - 20.0 * (phase - 0.3) / 0.4
            knee_angle = -60.0 + 50.0 * (phase - 0.3) / 0.4
            ankle_angle = 20.0 - 20.0 * (phase - 0.3) / 0.4
        else:
            # 落地阶段
            hip_angle = 10.0 - 15.0 * (phase - 0.7) / 0.3
            knee_angle = -10.0 - 5.0 * (phase - 0.7) / 0.3
            ankle_angle = 0.0 + 5.0 * (phase - 0.7) / 0.3
        
        self._apply_joint_pd_control(f"{side}_hip", np.radians(hip_angle))
        self._apply_joint_pd_control(f"{side}_knee", np.radians(knee_angle))
        self._apply_joint_pd_control(f"{side}_ankle", np.radians(ankle_angle))
    
    def _control_double_support(self):
        """控制双支撑阶段"""
        # 双支撑阶段，两腿都承重
        for side in ["left", "right"]:
            self._control_support_leg(side)
    
    def _apply_joint_pd_control(self, joint_name: str, target_angle: float, kp: float = 500.0, kd: float = 50.0):
        """应用关节PD控制"""
        try:
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor")
            
            if joint_id >= 0 and actuator_id >= 0:
                # 获取关节当前状态
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qvel_addr = self.model.jnt_dofadr[joint_id]
                
                current_angle = self.data.qpos[qpos_addr]
                current_vel = self.data.qvel[qvel_addr]
                
                # PD控制
                error = target_angle - current_angle
                torque = kp * error - kd * current_vel
                
                # 限制力矩
                max_torque = 100.0
                torque = np.clip(torque, -max_torque, max_torque)
                
                self.data.ctrl[actuator_id] = torque
        except Exception as e:
            print(f"Joint control error for {joint_name}: {e}")
    
    def _update_visualization(self):
        """更新可视化元素"""
        # 更新目标位置标记
        self._update_target_markers()
        
        # 更新状态颜色
        self._update_state_colors()
    
    def _update_target_markers(self):
        """更新目标位置标记"""
        # 更新左脚目标标记
        if self.left_foot_target is not None:
            try:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_target")
                if body_id >= 0:
                    self.model.body_pos[body_id] = self.left_foot_target
            except:
                pass
        
        # 更新右脚目标标记
        if self.right_foot_target is not None:
            try:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_target")
                if body_id >= 0:
                    self.model.body_pos[body_id] = self.right_foot_target
            except:
                pass
    
    def _update_state_colors(self):
        """更新状态颜色"""
        current_state = self.gait_scheduler.current_state
        
        # 更新躯干颜色
        if current_state in self.leg_state_colors:
            color = self.leg_state_colors[current_state]
            try:
                torso_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "torso")
                if torso_geom_id >= 0:
                    self.model.geom_rgba[torso_geom_id] = color
            except:
                pass
        
        # 更新腿部颜色以显示支撑/摆动状态
        if current_state == GaitState.LEFT_SUPPORT:
            # 左腿绿色（支撑），右腿红色（摆动）
            self._set_leg_color("left", [0.2, 1.0, 0.2, 1.0])
            self._set_leg_color("right", [1.0, 0.2, 0.2, 1.0])
        elif current_state == GaitState.RIGHT_SUPPORT:
            # 右腿绿色（支撑），左腿红色（摆动）
            self._set_leg_color("right", [0.2, 1.0, 0.2, 1.0])
            self._set_leg_color("left", [1.0, 0.2, 0.2, 1.0])
        else:
            # 双支撑或站立，两腿都是蓝色
            self._set_leg_color("left", [0.2, 0.2, 1.0, 1.0])
            self._set_leg_color("right", [0.2, 0.2, 1.0, 1.0])
    
    def _set_leg_color(self, side: str, color: List[float]):
        """设置腿部颜色"""
        leg_parts = [f"{side}_thigh", f"{side}_shin", f"{side}_foot"]
        for part in leg_parts:
            try:
                geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, part)
                if geom_id >= 0:
                    self.model.geom_rgba[geom_id] = color
            except:
                pass
    
    def get_state_text(self) -> str:
        """获取状态文本信息"""
        current_state = self.gait_scheduler.current_state
        
        # 获取足部力信息
        left_force = self.data_bus.get_end_effector_contact_force("left_foot") or 0.0
        right_force = self.data_bus.get_end_effector_contact_force("right_foot") or 0.0
        
        # 获取目标位置信息
        target_info = ""
        if self.left_foot_target is not None:
            target_info += f"  左脚: ({self.left_foot_target[0]:.2f}, {self.left_foot_target[1]:.2f}, {self.left_foot_target[2]:.2f})\n"
        if self.right_foot_target is not None:
            target_info += f"  右脚: ({self.right_foot_target[0]:.2f}, {self.right_foot_target[1]:.2f}, {self.right_foot_target[2]:.2f})\n"
        
        # 性能统计
        if len(self.state_history) > 1:
            avg_cycle_time = np.mean([h['duration'] for h in self.state_history[-10:]])
        else:
            avg_cycle_time = 0.0
        
        text = f"""步态状态可视化
────────────────────
当前状态: {current_state.value}
步态周期: #{self.gait_cycle_count}
────────────────────
足底力:
  左脚: {left_force:.1f}N
  右脚: {right_force:.1f}N
────────────────────
目标位置:
{target_info}────────────────────
性能统计:
  平均周期: {avg_cycle_time:.3f}s
  连续周期: {self.gait_cycle_count}
────────────────────
控制:
  空格键: 暂停/继续
  R: 重置
  S: 开始步行
  +/-: 调节速度
  ESC: 退出
"""
        return text
    
    def run_simulation(self):
        """运行仿真（使用基于OpenGL的渲染）"""
        print("启动步态可视化仿真...")
        
        # 创建渲染上下文
        from mujoco import viewer
        
        # 使用标准viewer
        with viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=True,
            show_right_ui=True
        ) as v:
            # 设置摄像机视角
            v.cam.distance = 3.0
            v.cam.elevation = -20
            v.cam.azimuth = 45
            
            print("仿真已启动。使用以下控制:")
            print("  空格键: 暂停/继续")
            print("  R: 重置")
            print("  S: 开始步行")
            print("  +/-: 调节速度")
            print("  ESC: 退出")
            
            # 主循环
            self.running = True
            last_update_time = time.time()
            
            # 开始步行
            self.gait_scheduler.start_walking()
            
            while v.is_running() and self.running:
                current_time = time.time()
                
                if not self.paused:
                    # 更新步态系统
                    if current_time - last_update_time >= self.dt / self.simulation_speed:
                        self.update_gait_system()
                        self._walking_control()
                        last_update_time = current_time
                    
                    # 前进仿真
                    mj.mj_step(self.model, self.data)
                
                # --- 可视化: 目标落脚点 vs 当前足端位置 ---
                with v.lock():
                    # 清空自定义场景
                    v.user_scn.ngeom = 0
                    geom_idx = 0

                    def _add_sphere(pos, rgba):
                        nonlocal geom_idx
                        if geom_idx >= len(v.user_scn.geoms):
                            return
                        mj.mjv_initGeom(
                            v.user_scn.geoms[geom_idx],
                            type=mj.mjtGeom.mjGEOM_SPHERE,
                            size=[0.025, 0, 0],
                            pos=pos,
                            mat=np.eye(3).flatten(),
                            rgba=rgba,
                        )
                        geom_idx += 1

                    # 目标位置: 半透明红/绿
                    if self.left_foot_target is not None:
                        _add_sphere(self.left_foot_target, [1.0, 0.0, 0.0, 0.4])
                    if self.right_foot_target is not None:
                        _add_sphere(self.right_foot_target, [0.0, 1.0, 0.0, 0.4])
                    # 当前足端真实位置: 实心蓝色
                    left_pos = self._get_foot_position("left")
                    right_pos = self._get_foot_position("right")
                    _add_sphere(left_pos, [0.0, 0.3, 1.0, 1.0])
                    _add_sphere(right_pos, [0.0, 0.3, 1.0, 1.0])

                    v.user_scn.ngeom = geom_idx

                # 更新状态文本（使用overlay）
                # 新版 MuJoCo viewer 的 Handle 可能不再提供 add_overlay 接口。
                try:
                    v.add_overlay(mj.mjtGridPos.mjGRID_TOPLEFT, "状态信息", self.get_state_text())
                except AttributeError:
                    pass
                
                # 同步渲染
                v.sync()
        
        print("仿真结束")
    
    def run_offscreen_render(self):
        """离屏渲染版本（适用于无显示器环境）"""
        print("启动离屏渲染模式...")
        
        # 创建离屏渲染器
        self.renderer = mj.Renderer(self.model)
        
        # 设置场景
        self.renderer.update_scene(self.data, camera="free")
        
        # 主循环
        self.running = True
        last_update_time = time.time()
        frame_count = 0
        
        # 开始步行
        self.gait_scheduler.start_walking()
        
        while self.running and frame_count < 1000:  # 限制帧数
            current_time = time.time()
            
            if current_time - last_update_time >= self.dt:
                # 更新步态系统
                self.update_gait_system()
                self._walking_control()
                
                # 前进仿真
                mj.mj_step(self.model, self.data)
                
                # 每100帧打印一次状态
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count}: State = {self.gait_scheduler.current_state.value}")
                
                last_update_time = current_time
                frame_count += 1
        
        print("离屏渲染完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="步态可视化仿真")
    parser.add_argument("--model", type=str, default="models/scene.xml",
                       help="模型文件路径")
    parser.add_argument("--offscreen", action="store_true",
                       help="离屏渲染模式")
    parser.add_argument("--test", action="store_true",
                       help="测试模式")
    
    args = parser.parse_args()
    
    try:
        # 创建仿真实例
        sim = GaitVisualizationSimulation(args.model)
        
        if args.test:
            # 测试模式：检查功能是否正常
            print("运行功能测试...")
            sim.gait_scheduler.start_walking()
            
            for i in range(5000):  # 5秒测试
                sim.update_gait_system()
                sim._walking_control()
                mj.mj_step(sim.model, sim.data)
                
                if i % 1000 == 0:
                    print(f"Test step {i}: State = {sim.gait_scheduler.current_state.value}")
            
            print(f"测试完成: {sim.gait_cycle_count} 个完整步态周期")
            
        elif args.offscreen:
            # 离屏渲染模式
            sim.run_offscreen_render()
        else:
            # GUI模式
            # 检查是否在macOS上运行
            if sys.platform == "darwin":
                print("\n注意: 在macOS上运行MuJoCo GUI需要使用特殊的启动方式。")
                print("如果遇到错误，请尝试以下命令:")
                print(f"  mjpython {__file__}")
                print("或使用离屏模式:")
                print(f"  python {__file__} --offscreen")
            
            sim.run_simulation()
            
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 