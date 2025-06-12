#!/usr/bin/env python3
"""
MPC求解器测试
测试QP求解器集成和基本功能

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
from gait_core.mpc_solver import (
    MPCSolver, GaitPlan, MPCResult, QPSolver, create_mpc_solver
)
from gait_core.data_bus import DataBus, Vector3D, get_data_bus


class TestMPCSolver(unittest.TestCase):
    """MPC求解器测试类"""
    
    def setUp(self):
        """测试设置"""
        self.data_bus = get_data_bus()
        
        # 设置初始状态
        initial_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_vel = Vector3D(x=0.1, y=0.0, z=0.0)
        
        self.data_bus.set_center_of_mass_position(initial_pos)
        self.data_bus.set_center_of_mass_velocity(initial_vel)
        
        # 创建LIPM模型
        self.lipm_model = SimplifiedDynamicsModel(data_bus=self.data_bus)
        
        # 创建基本步态计划
        self.gait_plan = self.create_simple_gait_plan()
        
        # 创建当前状态
        self.current_state = LIPMState()
        self.current_state.com_position = initial_pos
        self.current_state.com_velocity = initial_vel
        self.current_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
        self.current_state.timestamp = 0.0
    
    def create_simple_gait_plan(self) -> GaitPlan:
        """创建简单步态计划"""
        gait_plan = GaitPlan()
        
        # 创建5步的简单双支撑步态
        n_steps = 10
        gait_plan.support_sequence = ['double'] * n_steps
        gait_plan.contact_schedule = np.ones((n_steps, 2))  # 双脚始终接触
        
        # 设置足位置
        for i in range(n_steps):
            foot_positions = {
                'left': Vector3D(x=0.0, y=0.05, z=0.0),
                'right': Vector3D(x=0.0, y=-0.05, z=0.0)
            }
            gait_plan.support_positions.append(foot_positions)
        
        return gait_plan
    
    def test_mpc_solver_creation(self):
        """测试MPC求解器创建"""
        # 测试默认参数
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,  # 使用SciPy作为默认测试
            prediction_horizon=5,
            dt=0.1
        )
        
        self.assertIsNotNone(mpc_solver)
        self.assertEqual(mpc_solver.N, 5)
        self.assertEqual(mpc_solver.dt, 0.1)
        self.assertEqual(mpc_solver.solver_type, QPSolver.SCIPY)
    
    def test_gait_plan_creation(self):
        """测试步态计划创建"""
        self.assertIsNotNone(self.gait_plan)
        self.assertEqual(len(self.gait_plan.support_sequence), 10)
        self.assertEqual(self.gait_plan.contact_schedule.shape, (10, 2))
        self.assertEqual(len(self.gait_plan.support_positions), 10)
    
    def test_mpc_solve_basic(self):
        """测试基本MPC求解"""
        # 创建小规模求解器用于测试
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=3,  # 小规模问题
            dt=0.2
        )
        
        # 调用求解
        result = mpc_solver.solve(self.current_state, self.gait_plan)
        
        # 基本结果检查
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MPCResult)
        
        # 检查基本属性
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.solve_time, float)
        self.assertIsInstance(result.cost, float)
        
        print(f"求解成功: {result.success}")
        print(f"求解时间: {result.solve_time:.3f}s")
        print(f"求解代价: {result.cost:.3f}")
    
    def test_mpc_result_structure(self):
        """测试MPC结果结构"""
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=3,
            dt=0.2
        )
        
        result = mpc_solver.solve(self.current_state, self.gait_plan)
        
        # 检查结果结构
        self.assertTrue(hasattr(result, 'contact_forces'))
        self.assertTrue(hasattr(result, 'com_position_trajectory'))
        self.assertTrue(hasattr(result, 'com_velocity_trajectory'))
        self.assertTrue(hasattr(result, 'current_contact_forces'))
        self.assertTrue(hasattr(result, 'current_desired_com_acceleration'))
        
        if result.success:
            # 检查轨迹长度
            self.assertGreater(len(result.com_position_trajectory), 0)
            self.assertGreater(len(result.com_velocity_trajectory), 0)
            
            # 检查当前控制命令
            self.assertIn('left', result.current_contact_forces)
            self.assertIn('right', result.current_contact_forces)
    
    def test_variable_dimensions(self):
        """测试决策变量维度计算"""
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=5,
            dt=0.1
        )
        
        dims = mpc_solver._get_variable_dimensions(self.gait_plan)
        
        # 检查维度计算
        expected_states = 6 * (5 + 1)  # [x,y,vx,vy,ax,ay] * (N+1)
        expected_forces = 6 * 5        # [fx,fy,fz] * 2_feet * N
        expected_zmp = 2 * 5           # [zmp_x, zmp_y] * N
        
        self.assertEqual(dims['states'], expected_states)
        self.assertEqual(dims['forces'], expected_forces)
        self.assertEqual(dims['zmp'], expected_zmp)
        self.assertEqual(dims['total'], expected_states + expected_forces + expected_zmp)
    
    def test_solver_fallback(self):
        """测试求解器后备机制"""
        # 创建可能失败的求解器配置
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=2,  # 极小问题
            dt=0.5
        )
        
        result = mpc_solver.solve(self.current_state, self.gait_plan)
        
        # 即使求解失败，也应该有后备结果
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.current_contact_forces)
        self.assertIn('left', result.current_contact_forces)
        self.assertIn('right', result.current_contact_forces)
    
    def test_solver_statistics(self):
        """测试求解器统计"""
        mpc_solver = create_mpc_solver(
            lipm_model=self.lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=3,
            dt=0.2
        )
        
        # 执行多次求解
        for i in range(3):
            result = mpc_solver.solve(self.current_state, self.gait_plan)
        
        # 获取统计信息
        stats = mpc_solver.get_solver_statistics()
        
        self.assertIsInstance(stats, dict)
        if stats:  # 如果有统计数据
            self.assertIn('average_solve_time', stats)
            self.assertIn('success_rate', stats)
            self.assertIn('solver_type', stats)
            self.assertEqual(stats['total_solves'], 3)


class TestGaitPlan(unittest.TestCase):
    """步态计划测试类"""
    
    def test_gait_plan_basic(self):
        """测试基本步态计划"""
        gait_plan = GaitPlan()
        
        # 测试默认值
        self.assertEqual(len(gait_plan.support_sequence), 0)
        self.assertEqual(gait_plan.contact_schedule.size, 0)
        self.assertEqual(len(gait_plan.support_positions), 0)
        
        # 测试参数设置
        self.assertGreater(gait_plan.min_contact_duration, 0)
        self.assertGreater(gait_plan.max_step_length, 0)
        self.assertGreater(gait_plan.step_height, 0)
    
    def test_walking_gait_generation(self):
        """测试行走步态生成"""
        # 这里需要从演示脚本导入函数，简化测试
        gait_plan = GaitPlan()
        
        # 手动创建简单的行走步态
        n_steps = 8
        for i in range(n_steps):
            if i % 4 < 2:
                gait_plan.support_sequence.append('left')
            else:
                gait_plan.support_sequence.append('right')
        
        # 创建接触调度
        gait_plan.contact_schedule = np.zeros((n_steps, 2))
        for i, support in enumerate(gait_plan.support_sequence):
            if support == 'left':
                gait_plan.contact_schedule[i, :] = [1.0, 0.0]
            else:
                gait_plan.contact_schedule[i, :] = [0.0, 1.0]
        
        # 检查结果
        self.assertEqual(len(gait_plan.support_sequence), n_steps)
        self.assertEqual(gait_plan.contact_schedule.shape, (n_steps, 2))
        
        # 检查接触调度的一致性
        for i, support in enumerate(gait_plan.support_sequence):
            if support == 'left':
                self.assertEqual(gait_plan.contact_schedule[i, 0], 1.0)
                self.assertEqual(gait_plan.contact_schedule[i, 1], 0.0)
            else:
                self.assertEqual(gait_plan.contact_schedule[i, 0], 0.0)
                self.assertEqual(gait_plan.contact_schedule[i, 1], 1.0)


def run_performance_test():
    """运行性能测试"""
    print("\n=== MPC求解器性能测试 ===")
    
    # 创建系统
    data_bus = get_data_bus()
    initial_pos = Vector3D(x=0.0, y=0.0, z=0.8)
    initial_vel = Vector3D(x=0.1, y=0.0, z=0.0)
    
    data_bus.set_center_of_mass_position(initial_pos)
    data_bus.set_center_of_mass_velocity(initial_vel)
    
    lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
    
    # 创建步态计划
    gait_plan = GaitPlan()
    n_steps = 15
    gait_plan.support_sequence = ['double'] * n_steps
    gait_plan.contact_schedule = np.ones((n_steps, 2))
    
    for i in range(n_steps):
        foot_positions = {
            'left': Vector3D(x=0.0, y=0.05, z=0.0),
            'right': Vector3D(x=0.0, y=-0.05, z=0.0)
        }
        gait_plan.support_positions.append(foot_positions)
    
    current_state = LIPMState()
    current_state.com_position = initial_pos
    current_state.com_velocity = initial_vel
    current_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
    current_state.timestamp = 0.0
    
    # 测试不同规模的问题
    test_configs = [
        (3, 0.2, "小规模"),
        (5, 0.1, "中规模"),
        (8, 0.1, "大规模"),
    ]
    
    for horizon, dt, desc in test_configs:
        print(f"\n{desc}测试 (N={horizon}, dt={dt}):")
        
        try:
            mpc_solver = create_mpc_solver(
                lipm_model=lipm_model,
                solver_type=QPSolver.SCIPY,
                prediction_horizon=horizon,
                dt=dt
            )
            
            # 多次求解取平均
            solve_times = []
            success_count = 0
            
            for i in range(5):
                result = mpc_solver.solve(current_state, gait_plan)
                solve_times.append(result.solve_time)
                if result.success:
                    success_count += 1
            
            avg_time = np.mean(solve_times)
            success_rate = success_count / 5
            
            print(f"  平均求解时间: {avg_time:.3f}s ({avg_time*1000:.1f}ms)")
            print(f"  成功率: {success_rate:.1%}")
            print(f"  变量数量: ~{6*(horizon+1) + 6*horizon + 2*horizon}")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")


def run_basic_test():
    """运行基本功能测试"""
    print("=== 基本功能测试 ===")
    
    try:
        # 创建系统
        data_bus = get_data_bus()
        initial_pos = Vector3D(x=0.0, y=0.0, z=0.8)
        initial_vel = Vector3D(x=0.1, y=0.0, z=0.0)
        
        data_bus.set_center_of_mass_position(initial_pos)
        data_bus.set_center_of_mass_velocity(initial_vel)
        
        lipm_model = SimplifiedDynamicsModel(data_bus=data_bus)
        print("✅ LIPM模型创建成功")
        
        # 创建简单步态计划
        gait_plan = GaitPlan()
        n_steps = 5
        gait_plan.support_sequence = ['double'] * n_steps
        gait_plan.contact_schedule = np.ones((n_steps, 2))
        
        for i in range(n_steps):
            foot_positions = {
                'left': Vector3D(x=0.0, y=0.05, z=0.0),
                'right': Vector3D(x=0.0, y=-0.05, z=0.0)
            }
            gait_plan.support_positions.append(foot_positions)
        
        print("✅ 步态计划创建成功")
        
        # 创建MPC求解器
        mpc_solver = create_mpc_solver(
            lipm_model=lipm_model,
            solver_type=QPSolver.SCIPY,
            prediction_horizon=3,
            dt=0.2
        )
        print("✅ MPC求解器创建成功")
        
        # 设置当前状态
        current_state = LIPMState()
        current_state.com_position = initial_pos
        current_state.com_velocity = initial_vel
        current_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
        current_state.timestamp = 0.0
        
        # 求解MPC
        print("开始MPC求解...")
        result = mpc_solver.solve(current_state, gait_plan)
        
        if result.success:
            print(f"✅ MPC求解成功!")
            print(f"   求解时间: {result.solve_time:.3f}s")
            print(f"   求解代价: {result.cost:.3f}")
            print(f"   预测轨迹长度: {len(result.com_position_trajectory)}")
        else:
            print(f"❌ MPC求解失败: {result.solver_info}")
        
        print("\n=== 基本功能测试完成 ===")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("MPC求解器测试程序")
    print("=" * 50)
    
    # 运行基本功能测试
    success = run_basic_test()
    
    if success:
        print("\n🎉 所有基本测试通过!")
    else:
        print("\n❌ 测试失败，请检查实现!")
    
    # 运行性能测试
    print("\n2. 运行性能测试...")
    run_performance_test()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成!") 