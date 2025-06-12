#!/usr/bin/env python3
"""
MPC控制器测试
验证模型预测控制器的功能和正确性

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

from gait_core.simplified_dynamics import SimplifiedDynamicsModel, LIPMState
from gait_core.mpc_controller import (
    MPCController, MPCParameters, MPCReference, MPCMode, MPCSolution
)
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


class TestMPCParameters(unittest.TestCase):
    """测试MPC参数类"""
    
    def test_parameter_initialization(self):
        """测试参数初始化"""
        params = MPCParameters()
        
        # 检查默认值
        self.assertEqual(params.prediction_horizon, 20)
        self.assertEqual(params.control_horizon, 10)
        self.assertEqual(params.dt, 0.1)
        
        # 检查权重矩阵
        self.assertEqual(params.Q_position.shape, (2, 2))
        self.assertEqual(params.R_force.shape, (2, 2))
        self.assertTrue(np.all(np.diag(params.Q_position) > 0))
        self.assertTrue(np.all(np.diag(params.R_force) > 0))
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        params = MPCParameters()
        params.prediction_horizon = 30
        params.control_horizon = 15
        params.dt = 0.05
        params.max_force = 100.0
        
        self.assertEqual(params.prediction_horizon, 30)
        self.assertEqual(params.control_horizon, 15)
        self.assertEqual(params.dt, 0.05)
        self.assertEqual(params.max_force, 100.0)


class TestMPCReference(unittest.TestCase):
    """测试MPC参考轨迹类"""
    
    def test_reference_initialization(self):
        """测试参考轨迹初始化"""
        ref = MPCReference()
        
        self.assertEqual(len(ref.com_position_ref), 0)
        self.assertEqual(len(ref.com_velocity_ref), 0)
        self.assertEqual(len(ref.footstep_ref), 0)
        self.assertEqual(len(ref.support_leg_sequence), 0)
    
    def test_reference_population(self):
        """测试参考轨迹填充"""
        ref = MPCReference()
        
        # 添加参考位置
        ref.com_position_ref.append(Vector3D(x=0.1, y=0.0, z=0.8))
        ref.com_position_ref.append(Vector3D(x=0.2, y=0.0, z=0.8))
        
        # 添加参考速度
        ref.com_velocity_ref.append(Vector3D(x=0.1, y=0.0, z=0.0))
        ref.com_velocity_ref.append(Vector3D(x=0.1, y=0.0, z=0.0))
        
        self.assertEqual(len(ref.com_position_ref), 2)
        self.assertEqual(len(ref.com_velocity_ref), 2)
        self.assertEqual(ref.com_position_ref[0].x, 0.1)
        self.assertEqual(ref.com_position_ref[1].x, 0.2)


class TestMPCController(unittest.TestCase):
    """测试MPC控制器类"""
    
    def setUp(self):
        """测试设置"""
        # 创建数据总线和LIPM模型
        self.data_bus = DataBus()
        
        initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(initial_com_pos)
        self.data_bus.set_center_of_mass_velocity(initial_com_vel)
        
        self.lipm_model = SimplifiedDynamicsModel(data_bus=self.data_bus)
        
        # 创建MPC参数
        self.mpc_params = MPCParameters()
        self.mpc_params.prediction_horizon = 10  # 较短的时域用于测试
        self.mpc_params.control_horizon = 5
        self.mpc_params.dt = 0.1
        
        # 创建MPC控制器
        self.mpc_controller = MPCController(self.lipm_model, self.mpc_params, self.data_bus)
    
    def test_controller_initialization(self):
        """测试控制器初始化"""
        self.assertEqual(self.mpc_controller.lipm_model, self.lipm_model)
        self.assertEqual(self.mpc_controller.params, self.mpc_params)
        self.assertEqual(self.mpc_controller.data_bus, self.data_bus)
        self.assertEqual(self.mpc_controller.control_mode, MPCMode.COMBINED_CONTROL)
    
    def test_reference_setting(self):
        """测试参考轨迹设置"""
        reference = MPCReference()
        
        # 生成简单参考轨迹
        for i in range(10):
            t = i * 0.1
            pos = Vector3D(x=0.1 * t, y=0.0, z=0.8)
            vel = Vector3D(x=0.1, y=0.0, z=0.0)
            reference.com_position_ref.append(pos)
            reference.com_velocity_ref.append(vel)
        
        self.mpc_controller.set_reference_trajectory(reference)
        
        self.assertEqual(len(self.mpc_controller.reference.com_position_ref), 10)
        self.assertEqual(len(self.mpc_controller.reference.com_velocity_ref), 10)
    
    def test_walking_reference_generation(self):
        """测试行走参考轨迹生成"""
        target_velocity = Vector3D(x=0.2, y=0.0, z=0.0)
        
        reference = self.mpc_controller.generate_walking_reference(
            target_velocity=target_velocity,
            step_length=0.15,
            step_width=0.1,
            step_duration=0.6
        )
        
        # 检查生成的参考轨迹
        self.assertGreater(len(reference.com_position_ref), 0)
        self.assertGreater(len(reference.com_velocity_ref), 0)
        self.assertGreater(len(reference.footstep_ref), 0)
        
        # 检查速度一致性
        for vel_ref in reference.com_velocity_ref:
            self.assertAlmostEqual(vel_ref.x, target_velocity.x, places=6)
            self.assertAlmostEqual(vel_ref.y, target_velocity.y, places=6)
    
    def test_control_mode_switching(self):
        """测试控制模式切换"""
        # 测试力控制模式
        self.mpc_controller.control_mode = MPCMode.FORCE_CONTROL
        self.assertEqual(self.mpc_controller.control_mode, MPCMode.FORCE_CONTROL)
        
        # 测试步态控制模式
        self.mpc_controller.control_mode = MPCMode.FOOTSTEP_CONTROL
        self.assertEqual(self.mpc_controller.control_mode, MPCMode.FOOTSTEP_CONTROL)
        
        # 测试组合控制模式
        self.mpc_controller.control_mode = MPCMode.COMBINED_CONTROL
        self.assertEqual(self.mpc_controller.control_mode, MPCMode.COMBINED_CONTROL)
    
    def test_fallback_solution(self):
        """测试后备解决方案"""
        current_state = LIPMState()
        current_state.com_position = Vector3D(x=0.0, y=0.0, z=0.8)
        current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        
        fallback_solution = self.mpc_controller._get_fallback_solution(current_state)
        
        self.assertIsInstance(fallback_solution, MPCSolution)
        self.assertFalse(fallback_solution.success)
        self.assertEqual(fallback_solution.solver_status, "Fallback")
        self.assertEqual(fallback_solution.cost, float('inf'))
    
    def test_trajectory_simulation(self):
        """测试轨迹模拟"""
        initial_state = LIPMState()
        initial_state.com_position = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_state.com_velocity = Vector3D(x=0.1, y=0.0, z=0.0)
        
        # 创建控制输入
        forces = [Vector3D(x=5.0, y=0.0, z=0.0) for _ in range(5)]
        footsteps = [Vector3D(x=0.0, y=0.0, z=0.0) for _ in range(5)]
        
        trajectory = self.mpc_controller._simulate_trajectory(
            initial_state, forces, footsteps
        )
        
        # 检查轨迹长度
        expected_length = self.mpc_params.prediction_horizon + 1
        self.assertEqual(len(trajectory), expected_length)
        
        # 检查轨迹连续性
        for i in range(1, len(trajectory)):
            # 位置应该是连续的
            dt = self.mpc_params.dt
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            
            # 简单的连续性检查（位置变化不应该过大）
            dx = abs(curr_state.com_position.x - prev_state.com_position.x)
            dy = abs(curr_state.com_position.y - prev_state.com_position.y)
            
            self.assertLess(dx, 0.5)  # 位置变化应该合理
            self.assertLess(dy, 0.5)
    
    def test_constraint_evaluation(self):
        """测试约束评估"""
        # 创建测试状态
        test_state = LIPMState()
        test_state.com_position = Vector3D(x=0.3, y=0.0, z=0.8)  # 接近边界
        test_state.com_velocity = Vector3D(x=0.8, y=0.0, z=0.0)
        
        # 创建测试力
        test_force = Vector3D(x=40.0, y=30.0, z=0.0)  # 接近力限制
        
        # 使用较严格的约束参数
        strict_params = MPCParameters()
        strict_params.max_force = 50.0
        strict_params.com_position_bounds = (-0.5, 0.5)
        strict_params.com_velocity_bounds = (-1.0, 1.0)
        
        # 导入约束检查函数
        from examples.mpc_demo import check_constraints
        
        violations = check_constraints(test_state, test_force, strict_params)
        
        # 检查约束违反检测
        force_magnitude = np.sqrt(test_force.x**2 + test_force.y**2)
        expected_force_violation = force_magnitude > strict_params.max_force
        
        self.assertEqual(violations['force'], expected_force_violation)
        self.assertFalse(violations['position'])  # 位置应该在界内
        self.assertFalse(violations['velocity'])  # 速度应该在界内
    
    def test_mpc_solve_basic(self):
        """测试基本的MPC求解"""
        # 设置简单的参考轨迹
        reference = MPCReference()
        
        for i in range(self.mpc_params.prediction_horizon):
            t = i * self.mpc_params.dt
            pos = Vector3D(x=0.1 * np.sin(t), y=0.0, z=0.8)
            vel = Vector3D(x=0.1 * np.cos(t), y=0.0, z=0.0)
            reference.com_position_ref.append(pos)
            reference.com_velocity_ref.append(vel)
        
        self.mpc_controller.set_reference_trajectory(reference)
        
        # 设置当前状态
        current_state = LIPMState()
        current_state.com_position = Vector3D(x=0.0, y=0.0, z=0.8)
        current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        current_state.timestamp = 0.0
        
        # 设置控制模式为力控制（更简单）
        self.mpc_controller.control_mode = MPCMode.FORCE_CONTROL
        
        # 求解MPC
        solution = self.mpc_controller.solve_mpc(current_state)
        
        # 检查解的基本属性
        self.assertIsInstance(solution, MPCSolution)
        self.assertGreaterEqual(solution.solve_time, 0.0)
        
        # 如果求解成功，检查解的合理性
        if solution.success:
            self.assertLess(solution.cost, float('inf'))
            if solution.optimal_forces:
                self.assertGreater(len(solution.optimal_forces), 0)
            if solution.predicted_trajectory:
                self.assertGreater(len(solution.predicted_trajectory), 0)
    
    def test_performance_statistics(self):
        """测试性能统计"""
        # 运行几次求解以生成统计数据
        current_state = LIPMState()
        current_state.com_position = Vector3D(x=0.0, y=0.0, z=0.8)
        current_state.com_velocity = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.mpc_controller.control_mode = MPCMode.FORCE_CONTROL
        
        # 生成简单参考
        reference = MPCReference()
        for i in range(5):
            reference.com_position_ref.append(Vector3D(x=0.0, y=0.0, z=0.8))
            reference.com_velocity_ref.append(Vector3D(x=0.0, y=0.0, z=0.0))
        
        self.mpc_controller.set_reference_trajectory(reference)
        
        # 运行多次求解
        for _ in range(3):
            solution = self.mpc_controller.solve_mpc(current_state)
        
        # 检查统计数据
        stats = self.mpc_controller.get_performance_stats()
        
        if stats:  # 如果有统计数据
            self.assertIn('average_solve_time', stats)
            self.assertIn('total_solves', stats)
            self.assertEqual(stats['total_solves'], 3)
            self.assertGreaterEqual(stats['average_solve_time'], 0.0)


class TestMPCIntegration(unittest.TestCase):
    """测试MPC与LIPM模型的集成"""
    
    def setUp(self):
        """测试设置"""
        self.data_bus = DataBus()
        
        initial_com_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_com_vel = Vector3D(x=0.0, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(initial_com_pos)
        self.data_bus.set_center_of_mass_velocity(initial_com_vel)
        
        self.lipm_model = SimplifiedDynamicsModel(data_bus=self.data_bus)
        
        # 简化的MPC参数
        self.mpc_params = MPCParameters()
        self.mpc_params.prediction_horizon = 8
        self.mpc_params.control_horizon = 4
        self.mpc_params.dt = 0.1
        
        self.mpc_controller = MPCController(self.lipm_model, self.mpc_params)
    
    def test_update_control_loop(self):
        """测试控制更新循环"""
        # 设置简单参考
        reference = MPCReference()
        for i in range(8):
            reference.com_position_ref.append(Vector3D(x=0.05 * i, y=0.0, z=0.8))
            reference.com_velocity_ref.append(Vector3D(x=0.05, y=0.0, z=0.0))
        
        self.mpc_controller.set_reference_trajectory(reference)
        self.mpc_controller.control_mode = MPCMode.FORCE_CONTROL
        
        # 运行几步控制循环
        dt = 0.1
        for step in range(3):
            solution = self.mpc_controller.update_control(dt)
            
            # 检查控制更新的基本功能
            self.assertIsInstance(solution, MPCSolution)
            
            # 检查LIPM模型状态是否更新
            current_pos = self.lipm_model.current_state.com_position
            self.assertIsInstance(current_pos.x, float)
            self.assertIsInstance(current_pos.y, float)
    
    def test_data_bus_integration(self):
        """测试与数据总线的集成"""
        # 更新数据总线中的状态
        new_pos = Vector3D(x=0.1, y=0.05, z=0.85)
        new_vel = Vector3D(x=0.2, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(new_pos)
        self.data_bus.set_center_of_mass_velocity(new_vel)
        
        # 更新MPC控制器
        dt = 0.1
        solution = self.mpc_controller.update_control(dt)
        
        # 检查LIPM模型是否从数据总线读取了新状态
        # （注意：update_control会先从数据总线读取状态）
        current_state = self.lipm_model.current_state
        
        # 由于模型会进行一步预测，位置可能略有变化，但应该接近初始值
        self.assertAlmostEqual(current_state.com_position.z, 0.85, places=1)


if __name__ == '__main__':
    # 设置测试环境
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 运行测试
    unittest.main(verbosity=2) 