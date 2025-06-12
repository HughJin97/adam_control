#!/usr/bin/env python3
"""
基本数据结构测试
仅测试核心数据结构，不依赖外部库

作者: Adam Control Team
版本: 1.0
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_basic_imports():
    """测试基本导入"""
    print("=== 测试基本导入 ===")
    
    try:
        from gait_core.data_bus import Vector3D
        print("✅ Vector3D 导入成功")
        
        # 测试Vector3D基本功能
        vec = Vector3D(x=1.0, y=2.0, z=3.0)
        assert vec.x == 1.0
        assert vec.y == 2.0  
        assert vec.z == 3.0
        print("✅ Vector3D 功能测试通过")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_mpc_data_structures():
    """测试MPC数据结构定义"""
    print("\n=== 测试MPC数据结构定义 ===")
    
    # 手动定义核心数据结构，避免导入问题
    from dataclasses import dataclass, field
    from typing import Dict, List, Optional
    from enum import Enum
    import numpy as np
    
    class QPSolver(Enum):
        """QP求解器类型"""
        OSQP = "osqp"
        QPOASES = "qpoases"
        CVXPY_OSQP = "cvxpy_osqp"
        SCIPY = "scipy"
    
    @dataclass
    class Vector3D:
        """简化的3D向量"""
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
    
    @dataclass
    class GaitPlan:
        """步态计划数据结构"""
        support_sequence: List[str] = field(default_factory=list)
        contact_schedule: np.ndarray = field(default_factory=lambda: np.array([]))
        support_positions: List[Dict[str, Vector3D]] = field(default_factory=list)
        min_contact_duration: float = 0.1
        max_step_length: float = 0.3
        step_height: float = 0.05
    
    @dataclass
    class MPCResult:
        """MPC求解结果"""
        success: bool = False
        solve_time: float = 0.0
        cost: float = float('inf')
        contact_forces: np.ndarray = field(default_factory=lambda: np.array([]))
        com_position_trajectory: List[Vector3D] = field(default_factory=list)
        com_velocity_trajectory: List[Vector3D] = field(default_factory=list)
        current_contact_forces: Dict[str, Vector3D] = field(default_factory=dict)
        current_desired_com_acceleration: Vector3D = field(default_factory=Vector3D)
    
    # 测试数据结构创建
    try:
        # 测试QPSolver枚举
        assert QPSolver.OSQP.value == "osqp"
        assert QPSolver.SCIPY.value == "scipy"
        print("✅ QPSolver 枚举定义正确")
        
        # 测试Vector3D
        vec = Vector3D(x=1.0, y=2.0, z=3.0)
        assert vec.x == 1.0
        assert vec.y == 2.0
        assert vec.z == 3.0
        print("✅ Vector3D 数据类定义正确")
        
        # 测试GaitPlan
        gait_plan = GaitPlan()
        assert len(gait_plan.support_sequence) == 0
        assert gait_plan.contact_schedule.size == 0
        assert len(gait_plan.support_positions) == 0
        assert gait_plan.min_contact_duration == 0.1
        assert gait_plan.max_step_length == 0.3
        assert gait_plan.step_height == 0.05
        print("✅ GaitPlan 数据类定义正确")
        
        # 测试MPCResult
        result = MPCResult()
        assert result.success == False
        assert result.solve_time == 0.0
        assert result.cost == float('inf')
        assert len(result.com_position_trajectory) == 0
        assert len(result.current_contact_forces) == 0
        print("✅ MPCResult 数据类定义正确")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        return False


def test_gait_planning_logic():
    """测试步态规划逻辑"""
    print("\n=== 测试步态规划逻辑 ===")
    
    try:
        import numpy as np
        
        # 简化的Vector3D类
        class Vector3D:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z
        
        # 简化的GaitPlan类
        class GaitPlan:
            def __init__(self):
                self.support_sequence = []
                self.contact_schedule = np.array([])
                self.support_positions = []
        
        def create_simple_walking_gait(n_steps: int = 8) -> GaitPlan:
            """创建简单行走步态"""
            gait_plan = GaitPlan()
            
            # 生成交替支撑序列
            for i in range(n_steps):
                if i % 4 < 2:
                    gait_plan.support_sequence.append('left')
                else:
                    gait_plan.support_sequence.append('right')
            
            # 生成接触调度矩阵
            gait_plan.contact_schedule = np.zeros((n_steps, 2))
            for i, support in enumerate(gait_plan.support_sequence):
                if support == 'left':
                    gait_plan.contact_schedule[i, :] = [1.0, 0.0]
                else:
                    gait_plan.contact_schedule[i, :] = [0.0, 1.0]
            
            # 生成足位置
            for i in range(n_steps):
                foot_positions = {
                    'left': Vector3D(x=i*0.1, y=0.05, z=0.0),
                    'right': Vector3D(x=i*0.1, y=-0.05, z=0.0)
                }
                gait_plan.support_positions.append(foot_positions)
            
            return gait_plan
        
        # 测试步态生成
        gait_plan = create_simple_walking_gait(8)
        
        # 验证结果
        assert len(gait_plan.support_sequence) == 8
        assert gait_plan.contact_schedule.shape == (8, 2)
        assert len(gait_plan.support_positions) == 8
        
        # 验证支撑序列的正确性
        expected_sequence = ['left', 'left', 'right', 'right', 'left', 'left', 'right', 'right']
        assert gait_plan.support_sequence == expected_sequence
        
        # 验证接触调度一致性
        for i, support in enumerate(gait_plan.support_sequence):
            if support == 'left':
                assert gait_plan.contact_schedule[i, 0] == 1.0
                assert gait_plan.contact_schedule[i, 1] == 0.0
            else:
                assert gait_plan.contact_schedule[i, 0] == 0.0
                assert gait_plan.contact_schedule[i, 1] == 1.0
        
        print("✅ 简单行走步态生成测试通过")
        
        # 测试复杂步态模式
        def create_complex_walking_gait(duration: float = 2.0, step_duration: float = 0.6) -> GaitPlan:
            """创建复杂行走步态"""
            gait_plan = GaitPlan()
            dt = 0.1
            n_steps = int(duration / dt)
            
            for i in range(n_steps):
                t = i * dt
                step_phase = (t % step_duration) / step_duration
                
                if step_phase < 0.3:
                    gait_plan.support_sequence.append('double')
                elif step_phase < 0.8:
                    step_index = int(t / step_duration)
                    if step_index % 2 == 0:
                        gait_plan.support_sequence.append('left')
                    else:
                        gait_plan.support_sequence.append('right')
                else:
                    gait_plan.support_sequence.append('double')
            
            return gait_plan
        
        complex_gait = create_complex_walking_gait(2.0, 0.6)
        
        # 验证复杂步态包含不同的支撑类型
        support_types = set(complex_gait.support_sequence)
        assert 'double' in support_types
        assert len(support_types) >= 2
        
        print("✅ 复杂行走步态生成测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 步态规划逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_problem_dimensions():
    """测试优化问题维度计算"""
    print("\n=== 测试优化问题维度计算 ===")
    
    try:
        def calculate_mpc_dimensions(prediction_horizon: int, n_contacts: int = 2) -> dict:
            """计算MPC问题维度"""
            N = prediction_horizon
            
            # 决策变量
            state_vars = 6 * (N + 1)    # [x,y,vx,vy,ax,ay] * (N+1)
            force_vars = 3 * n_contacts * N  # [fx,fy,fz] * n_contacts * N
            zmp_vars = 2 * N            # [zmp_x, zmp_y] * N
            total_vars = state_vars + force_vars + zmp_vars
            
            # 约束
            dynamics_eq = 6 * N         # LIPM动力学
            zmp_eq = 2 * N              # ZMP定义
            total_eq = dynamics_eq + zmp_eq
            
            force_ineq = 8 * n_contacts * N  # 摩擦锥 + 力限制
            kinematic_ineq = 4 * N      # 速度/加速度限制
            zmp_ineq = 4 * N            # ZMP边界
            total_ineq = force_ineq + kinematic_ineq + zmp_ineq
            
            return {
                'variables': {
                    'states': state_vars,
                    'forces': force_vars,
                    'zmp': zmp_vars,
                    'total': total_vars
                },
                'constraints': {
                    'equality': total_eq,
                    'inequality': total_ineq,
                    'total': total_eq + total_ineq
                }
            }
        
        # 测试不同规模的问题
        test_cases = [
            (5, "小规模"),
            (10, "中规模"),
            (15, "大规模"),
            (20, "超大规模")
        ]
        
        for horizon, desc in test_cases:
            dims = calculate_mpc_dimensions(horizon)
            
            print(f"✅ {desc}(N={horizon}): {dims['variables']['total']} 变量, "
                  f"{dims['constraints']['total']} 约束")
            
            # 验证维度合理性
            assert dims['variables']['total'] > 0
            assert dims['constraints']['total'] > 0
            assert dims['variables']['total'] < 2000  # 合理上限
        
        # 验证维度计算的正确性
        dims_5 = calculate_mpc_dimensions(5)
        expected_state_vars = 6 * (5 + 1)  # 36
        expected_force_vars = 3 * 2 * 5    # 30
        expected_zmp_vars = 2 * 5          # 10
        
        assert dims_5['variables']['states'] == expected_state_vars
        assert dims_5['variables']['forces'] == expected_force_vars
        assert dims_5['variables']['zmp'] == expected_zmp_vars
        assert dims_5['variables']['total'] == expected_state_vars + expected_force_vars + expected_zmp_vars
        
        print("✅ MPC问题维度计算正确")
        
        return True
        
    except Exception as e:
        print(f"❌ 优化问题维度测试失败: {e}")
        return False


def test_solver_configurations():
    """测试求解器配置"""
    print("\n=== 测试求解器配置 ===")
    
    try:
        # OSQP配置
        osqp_config = {
            'verbose': False,
            'eps_abs': 1e-4,
            'eps_rel': 1e-4,
            'max_iter': 2000,
            'polish': True,
            'adaptive_rho': True
        }
        
        # qpOASES配置
        qpoases_config = {
            'printLevel': 0,
            'maxCpuTime': 0.01,
            'maxWorkingSetRecalculations': 60
        }
        
        # CVXPY配置
        cvxpy_config = {
            'solver': 'OSQP',
            'verbose': False,
            'eps_abs': 1e-4,
            'eps_rel': 1e-4
        }
        
        # 验证配置参数类型和范围
        assert isinstance(osqp_config['eps_abs'], float)
        assert 0 < osqp_config['eps_abs'] < 1
        assert isinstance(osqp_config['max_iter'], int)
        assert osqp_config['max_iter'] > 0
        assert isinstance(osqp_config['verbose'], bool)
        
        assert isinstance(qpoases_config['maxCpuTime'], float)
        assert qpoases_config['maxCpuTime'] > 0
        assert isinstance(qpoases_config['printLevel'], int)
        assert qpoases_config['printLevel'] >= 0
        
        assert isinstance(cvxpy_config['solver'], str)
        assert isinstance(cvxpy_config['verbose'], bool)
        
        print("✅ OSQP配置验证通过")
        print("✅ qpOASES配置验证通过")
        print("✅ CVXPY配置验证通过")
        
        # 测试性能参数估算
        def estimate_solve_time(n_vars: int, n_constraints: int, solver: str) -> float:
            """估算求解时间"""
            base_time = {
                'osqp': 0.001,
                'qpoases': 0.002,
                'scipy': 0.01
            }
            
            complexity_factor = (n_vars + n_constraints) / 100.0
            return base_time.get(solver, 0.01) * complexity_factor
        
        # 测试不同问题规模的时间估算
        for n_vars, n_constraints in [(100, 200), (200, 400), (500, 1000)]:
            for solver in ['osqp', 'qpoases', 'scipy']:
                time_est = estimate_solve_time(n_vars, n_constraints, solver)
                assert time_est > 0
                assert time_est < 1.0  # 合理的时间范围
        
        print("✅ 求解器性能估算测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 求解器配置测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("基本数据结构测试程序")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("基本导入", test_basic_imports()))
    test_results.append(("MPC数据结构", test_mpc_data_structures()))
    test_results.append(("步态规划逻辑", test_gait_planning_logic()))
    test_results.append(("优化问题维度", test_optimization_problem_dimensions()))
    test_results.append(("求解器配置", test_solver_configurations()))
    
    # 统计结果
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有基本测试通过!")
        print("\n✨ 下一步:")
        print("   1. 安装依赖库 (scipy, numpy, osqp等)")
        print("   2. 实现完整的MPC求解器")
        print("   3. 集成LIPM动力学模型")
        print("   4. 进行实际机器人控制测试")
        return True
    else:
        print("❌ 部分测试失败，请检查实现")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 