#!/usr/bin/env python3
"""
简化动力学模型测试
验证LIPM模型的基本功能和数学正确性

作者: Adam Control Team
版本: 1.0
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from gait_core.simplified_dynamics import (
    SimplifiedDynamicsModel, 
    LIPMParameters, 
    LIPMState, 
    LIPMInput, 
    ControlMode
)
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


class TestLIPMParameters(unittest.TestCase):
    """测试LIPM参数类"""
    
    def test_parameter_initialization(self):
        """测试参数初始化"""
        params = LIPMParameters()
        
        # 检查默认值
        self.assertEqual(params.total_mass, 50.0)
        self.assertEqual(params.com_height, 0.8)
        self.assertEqual(params.gravity, 9.81)
        
        # 检查派生参数计算
        expected_omega = np.sqrt(params.gravity / params.com_height)
        self.assertAlmostEqual(params.natural_frequency, expected_omega, places=6)
        self.assertAlmostEqual(params.time_constant, 1.0 / expected_omega, places=6)
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        params = LIPMParameters(total_mass=60.0, com_height=1.0, gravity=9.8)
        
        self.assertEqual(params.total_mass, 60.0)
        self.assertEqual(params.com_height, 1.0)
        self.assertEqual(params.gravity, 9.8)
        
        expected_omega = np.sqrt(9.8 / 1.0)
        self.assertAlmostEqual(params.natural_frequency, expected_omega, places=6)


class TestLIPMState(unittest.TestCase):
    """测试LIPM状态类"""
    
    def test_state_initialization(self):
        """测试状态初始化"""
        state = LIPMState()
        
        # 检查Vector3D初始化
        self.assertEqual(state.com_position.x, 0.0)
        self.assertEqual(state.com_position.y, 0.0)
        self.assertEqual(state.com_position.z, 0.0)
        
        self.assertEqual(state.yaw, 0.0)
        self.assertEqual(state.yaw_rate, 0.0)
        self.assertEqual(state.timestamp, 0.0)
    
    def test_state_assignment(self):
        """测试状态赋值"""
        state = LIPMState()
        
        state.com_position = Vector3D(x=0.1, y=0.2, z=0.8)
        state.yaw = 0.5
        state.timestamp = 1.0
        
        self.assertEqual(state.com_position.x, 0.1)
        self.assertEqual(state.com_position.y, 0.2)
        self.assertEqual(state.com_position.z, 0.8)
        self.assertEqual(state.yaw, 0.5)
        self.assertEqual(state.timestamp, 1.0)


class TestSimplifiedDynamicsModel(unittest.TestCase):
    """测试简化动力学模型类"""
    
    def setUp(self):
        """测试设置"""
        # 创建干净的数据总线
        self.data_bus = DataBus()
        
        # 设置初始状态
        initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(initial_com_pos)
        self.data_bus.set_center_of_mass_velocity(initial_com_vel)
        
        # 创建模型
        self.model = SimplifiedDynamicsModel(data_bus=self.data_bus)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        # 检查参数
        self.assertIsInstance(self.model.parameters, LIPMParameters)
        self.assertIsInstance(self.model.current_state, LIPMState)
        self.assertIsInstance(self.model.input_command, LIPMInput)
        
        # 检查数据总线连接
        self.assertEqual(self.model.data_bus, self.data_bus)
    
    def test_support_foot_position_setting(self):
        """测试支撑足位置设置"""
        foot_pos = Vector3D(x=0.1, y=0.0, z=0.0)
        self.model.set_support_foot_position(foot_pos)
        
        self.assertEqual(self.model.input_command.support_foot_position.x, 0.1)
        self.assertEqual(self.model.input_command.support_foot_position.y, 0.0)
        self.assertEqual(self.model.input_command.support_foot_position.z, 0.0)
    
    def test_force_control_mode(self):
        """测试力控制模式"""
        force = Vector3D(x=10.0, y=5.0, z=0.0)
        self.model.set_foot_force_command(force)
        
        self.assertEqual(self.model.input_command.foot_force.x, 10.0)
        self.assertEqual(self.model.input_command.foot_force.y, 5.0)
        self.assertEqual(self.model.input_command.control_mode, ControlMode.FORCE_CONTROL)
    
    def test_footstep_control_mode(self):
        """测试步态控制模式"""
        footstep = Vector3D(x=0.2, y=0.1, z=0.0)
        self.model.set_next_footstep(footstep)
        
        self.assertEqual(self.model.input_command.next_footstep.x, 0.2)
        self.assertEqual(self.model.input_command.next_footstep.y, 0.1)
        self.assertEqual(self.model.input_command.control_mode, ControlMode.FOOTSTEP_CONTROL)
    
    def test_lipm_dynamics_equation(self):
        """测试LIPM动力学方程的正确性"""
        # 设置已知条件
        self.model.current_state.com_position = Vector3D(x=0.1, y=0.05, z=0.8)
        self.model.current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        dt = 0.01
        next_state = self.model.compute_com_dynamics(dt)
        
        # 验证LIMP方程: ẍ = (g/z_c) * (x - p_foot)
        g = self.model.parameters.gravity
        z_c = self.model.parameters.com_height
        x = 0.1
        y = 0.05
        p_foot_x = 0.0
        p_foot_y = 0.0
        
        expected_ax = (g / z_c) * (x - p_foot_x)
        expected_ay = (g / z_c) * (y - p_foot_y)
        
        self.assertAlmostEqual(next_state.com_acceleration.x, expected_ax, places=6)
        self.assertAlmostEqual(next_state.com_acceleration.y, expected_ay, places=6)
    
    def test_force_control_dynamics(self):
        """测试力控制下的动力学"""
        # 设置初始状态
        self.model.current_state.com_position = Vector3D(x=0.1, y=0.0, z=0.8)
        self.model.current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        # 施加控制力
        control_force = Vector3D(x=-20.0, y=0.0, z=0.0)  # 朝向支撑足的力
        self.model.set_foot_force_command(control_force)
        
        dt = 0.01
        next_state = self.model.compute_com_dynamics(dt)
        
        # 验证力控制方程: m*ẍ = mg*(x-p_foot)/z_c + f_x
        g = self.model.parameters.gravity
        z_c = self.model.parameters.com_height
        m = self.model.parameters.total_mass
        x = 0.1
        p_foot_x = 0.0
        fx = -20.0
        
        expected_ax = (g / z_c) * (x - p_foot_x) + fx / m
        
        self.assertAlmostEqual(next_state.com_acceleration.x, expected_ax, places=6)
    
    def test_numerical_integration(self):
        """测试数值积分的正确性"""
        # 设置初始条件
        initial_x = 0.1
        initial_vx = 0.2
        self.model.current_state.com_position = Vector3D(x=initial_x, y=0.0, z=0.8)
        self.model.current_state.com_velocity = Vector3D(x=initial_vx, y=0.0, z=0.0)
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        dt = 0.01
        next_state = self.model.compute_com_dynamics(dt)
        
        # 计算期望的加速度
        g = self.model.parameters.gravity
        z_c = self.model.parameters.com_height
        expected_ax = (g / z_c) * initial_x
        
        # 验证数值积分
        expected_next_x = initial_x + initial_vx * dt
        expected_next_vx = initial_vx + expected_ax * dt
        
        self.assertAlmostEqual(next_state.com_position.x, expected_next_x, places=6)
        self.assertAlmostEqual(next_state.com_velocity.x, expected_next_vx, places=6)
    
    def test_cop_calculation(self):
        """测试压力中心(COP)计算"""
        # 设置质心位置
        self.model.current_state.com_position = Vector3D(x=0.1, y=0.05, z=0.8)
        
        # 期望加速度
        desired_acc = Vector3D(x=-0.5, y=0.2, z=0.0)
        
        # 计算所需COP
        required_cop = self.model.compute_required_cop(desired_acc)
        
        # 验证COP计算: p_foot = x - (z_c/g) * ẍ
        z_c = self.model.parameters.com_height
        g = self.model.parameters.gravity
        x = 0.1
        y = 0.05
        ax = -0.5
        ay = 0.2
        
        expected_cop_x = x - (z_c / g) * ax
        expected_cop_y = y - (z_c / g) * ay
        
        self.assertAlmostEqual(required_cop.x, expected_cop_x, places=6)
        self.assertAlmostEqual(required_cop.y, expected_cop_y, places=6)
    
    def test_trajectory_prediction(self):
        """测试轨迹预测"""
        # 设置初始状态
        self.model.current_state.com_position = Vector3D(x=0.1, y=0.0, z=0.8)
        self.model.current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        # 预测轨迹
        time_horizon = 1.0
        dt = 0.1
        trajectory = self.model.predict_com_trajectory(time_horizon, dt)
        
        # 检查轨迹长度
        expected_steps = int(time_horizon / dt)
        self.assertEqual(len(trajectory), expected_steps)
        
        # 检查轨迹的单调性（应该趋向于支撑足）
        x_positions = [state.com_position.x for state in trajectory]
        
        # 由于质心在支撑足右侧，应该向左移动（减小）
        self.assertTrue(x_positions[0] > x_positions[-1])
    
    def test_quaternion_to_yaw_conversion(self):
        """测试四元数到偏航角的转换"""
        # 测试已知的四元数
        from gait_core.data_bus import Quaternion
        
        # 0度偏航角
        quat = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        yaw = self.model._quaternion_to_yaw(quat)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        
        # 90度偏航角
        quat = Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)  # 约45度 * 2 = 90度
        yaw = self.model._quaternion_to_yaw(quat)
        self.assertAlmostEqual(yaw, np.pi/2, places=2)
    
    def test_data_bus_integration(self):
        """测试与数据总线的集成"""
        # 设置数据总线中的状态
        new_com_pos = Vector3D(x=0.2, y=0.1, z=0.9)
        new_com_vel = Vector3D(x=0.1, y=0.05, z=0.0)
        
        self.data_bus.set_center_of_mass_position(new_com_pos)
        self.data_bus.set_center_of_mass_velocity(new_com_vel)
        
        # 更新模型状态
        self.model._update_state_from_data_bus()
        
        # 验证状态更新
        self.assertEqual(self.model.current_state.com_position.x, 0.2)
        self.assertEqual(self.model.current_state.com_position.y, 0.1)
        self.assertEqual(self.model.current_state.com_position.z, 0.9)
        self.assertEqual(self.model.current_state.com_velocity.x, 0.1)
        self.assertEqual(self.model.current_state.com_velocity.y, 0.05)


class TestLIPMStability(unittest.TestCase):
    """测试LIMP模型的稳定性"""
    
    def setUp(self):
        """测试设置"""
        self.data_bus = DataBus()
        self.model = SimplifiedDynamicsModel(data_bus=self.data_bus)
    
    def test_stability_analysis(self):
        """测试稳定性分析"""
        # 设置质心在支撑足正上方（稳定平衡点）
        stable_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        stable_vel = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(stable_pos)
        self.data_bus.set_center_of_mass_velocity(stable_vel)
        self.model._update_state_from_data_bus()
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        # 在平衡点，加速度应该为零
        dt = 0.01
        next_state = self.model.compute_com_dynamics(dt)
        
        self.assertAlmostEqual(next_state.com_acceleration.x, 0.0, places=6)
        self.assertAlmostEqual(next_state.com_acceleration.y, 0.0, places=6)
    
    def test_oscillation_behavior(self):
        """测试振荡行为"""
        # 设置初始偏移
        initial_pos = Vector3D(x=0.1, y=0.0, z=0.8)
        initial_vel = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(initial_pos)
        self.data_bus.set_center_of_mass_velocity(initial_vel)
        self.model._update_state_from_data_bus()
        
        support_foot = Vector3D(x=0.0, y=0.0, z=0.0)
        self.model.set_support_foot_position(support_foot)
        
        # 预测较长时间的轨迹
        time_horizon = 2.0
        dt = 0.01
        trajectory = self.model.predict_com_trajectory(time_horizon, dt)
        
        x_positions = [state.com_position.x for state in trajectory]
        
        # 检查是否存在振荡（位置应该在支撑足附近振荡）
        max_x = max(x_positions)
        min_x = min(x_positions)
        
        # 应该在支撑足两侧振荡
        self.assertTrue(max_x > 0.05)  # 右侧振幅
        self.assertTrue(min_x < -0.05)  # 左侧振幅


if __name__ == '__main__':
    # 设置测试环境
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 运行测试
    unittest.main(verbosity=2) 