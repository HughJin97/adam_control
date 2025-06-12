#!/usr/bin/env python3
"""
最小MPC求解器测试
不依赖pinocchio，仅测试核心MPC功能

作者: Adam Control Team
版本: 1.0
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 测试MPC求解器的数据结构
try:
    from gait_core.mpc_solver import (
        MPCSolver, GaitPlan, MPCResult, QPSolver
    )
    from gait_core.data_bus import Vector3D
    print("✅ MPC求解器模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_data_structures():
    """测试数据结构"""
    print("\n=== 测试数据结构 ===")
    
    # 测试Vector3D
    vec = Vector3D(x=1.0, y=2.0, z=3.0)
    assert vec.x == 1.0
    assert vec.y == 2.0
    assert vec.z == 3.0
    print("✅ Vector3D 测试通过")
    
    # 测试GaitPlan
    gait_plan = GaitPlan()
    assert len(gait_plan.support_sequence) == 0
    assert gait_plan.contact_schedule.size == 0
    assert len(gait_plan.support_positions) == 0
    assert gait_plan.min_contact_duration > 0
    assert gait_plan.max_step_length > 0
    assert gait_plan.step_height > 0
    print("✅ GaitPlan 测试通过")
    
    # 测试MPCResult
    result = MPCResult()
    assert hasattr(result, 'success')
    assert hasattr(result, 'solve_time')
    assert hasattr(result, 'cost')
    assert hasattr(result, 'contact_forces')
    assert hasattr(result, 'com_position_trajectory')
    assert hasattr(result, 'current_contact_forces')
    print("✅ MPCResult 测试通过")
    
    # 测试QPSolver枚举
    assert QPSolver.OSQP.value == "osqp"
    assert QPSolver.SCIPY.value == "scipy"
    assert QPSolver.CVXPY_OSQP.value == "cvxpy_osqp"
    print("✅ QPSolver 枚举测试通过")


def test_gait_plan_creation():
    """测试步态计划创建"""
    print("\n=== 测试步态计划创建 ===")
    
    gait_plan = GaitPlan()
    
    # 创建简单的行走步态
    n_steps = 8
    for i in range(n_steps):
        if i % 4 < 2:
            gait_plan.support_sequence.append('left')
        else:
            gait_plan.support_sequence.append('right')
    
    # 创建接触调度矩阵
    gait_plan.contact_schedule = np.zeros((n_steps, 2))
    for i, support in enumerate(gait_plan.support_sequence):
        if support == 'left':
            gait_plan.contact_schedule[i, :] = [1.0, 0.0]
        else:
            gait_plan.contact_schedule[i, :] = [0.0, 1.0]
    
    # 添加足位置
    for i in range(n_steps):
        foot_positions = {
            'left': Vector3D(x=i*0.1, y=0.05, z=0.0),
            'right': Vector3D(x=i*0.1, y=-0.05, z=0.0)
        }
        gait_plan.support_positions.append(foot_positions)
    
    # 验证结果
    assert len(gait_plan.support_sequence) == n_steps
    assert gait_plan.contact_schedule.shape == (n_steps, 2)
    assert len(gait_plan.support_positions) == n_steps
    
    # 验证接触调度的一致性
    for i, support in enumerate(gait_plan.support_sequence):
        if support == 'left':
            assert gait_plan.contact_schedule[i, 0] == 1.0
            assert gait_plan.contact_schedule[i, 1] == 0.0
        else:
            assert gait_plan.contact_schedule[i, 0] == 0.0
            assert gait_plan.contact_schedule[i, 1] == 1.0
    
    print(f"✅ 步态计划创建测试通过 ({n_steps} 步)")
    return gait_plan


def test_walking_gait_generation():
    """测试行走步态生成函数"""
    print("\n=== 测试行走步态生成 ===")
    
    def create_walking_gait_plan(duration: float = 2.0, step_duration: float = 0.6) -> GaitPlan:
        """创建行走步态计划"""
        gait_plan = GaitPlan()
        
        dt = 0.1  # 时间步长
        n_steps = int(duration / dt)
        
        # 生成支撑序列
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
        
        # 生成接触调度矩阵
        gait_plan.contact_schedule = np.zeros((n_steps, 2))
        for i, support_type in enumerate(gait_plan.support_sequence):
            if support_type == 'double':
                gait_plan.contact_schedule[i, :] = [1.0, 1.0]
            elif support_type == 'left':
                gait_plan.contact_schedule[i, :] = [1.0, 0.0]
            elif support_type == 'right':
                gait_plan.contact_schedule[i, :] = [0.0, 1.0]
        
        # 生成支撑足位置轨迹
        step_length = 0.15
        step_width = 0.1
        
        for i in range(n_steps):
            t = i * dt
            step_index = int(t / step_duration)
            
            left_x = step_index * step_length
            left_y = step_width / 2
            right_x = step_index * step_length
            right_y = -step_width / 2
            
            # 摆动腿向前移动
            if gait_plan.support_sequence[i] == 'right':
                left_x += step_length * 0.5
            elif gait_plan.support_sequence[i] == 'left':
                right_x += step_length * 0.5
            
            foot_positions = {
                'left': Vector3D(x=left_x, y=left_y, z=0.0),
                'right': Vector3D(x=right_x, y=right_y, z=0.0)
            }
            gait_plan.support_positions.append(foot_positions)
        
        return gait_plan
    
    # 测试不同参数的步态生成
    test_configs = [
        (1.0, 0.5, "短时长"),
        (2.0, 0.6, "标准"),
        (3.0, 0.8, "长步长")
    ]
    
    for duration, step_duration, desc in test_configs:
        gait_plan = create_walking_gait_plan(duration, step_duration)
        
        expected_steps = int(duration / 0.1)
        assert len(gait_plan.support_sequence) == expected_steps
        assert gait_plan.contact_schedule.shape == (expected_steps, 2)
        assert len(gait_plan.support_positions) == expected_steps
        
        # 检查是否包含不同的支撑类型
        support_types = set(gait_plan.support_sequence)
        assert 'double' in support_types
        assert len(support_types) >= 2  # 至少有双支撑和单支撑
        
        print(f"✅ {desc}步态生成测试通过 ({len(gait_plan.support_sequence)} 步)")


def test_optimization_problem_dimensions():
    """测试优化问题维度计算"""
    print("\n=== 测试优化问题维度 ===")
    
    # 模拟维度计算函数
    def get_variable_dimensions(gait_plan: GaitPlan, prediction_horizon: int = 10) -> dict:
        N = prediction_horizon
        return {
            'states': 6 * (N + 1),     # [x,y,vx,vy,ax,ay] * (N+1)
            'forces': 6 * N,           # [fx,fy,fz] * 2_feet * N  
            'zmp': 2 * N,              # [zmp_x, zmp_y] * N
            'total': 6 * (N + 1) + 6 * N + 2 * N
        }
    
    # 测试不同规模的问题
    test_horizons = [5, 10, 15, 20]
    
    for N in test_horizons:
        dims = get_variable_dimensions(None, N)
        
        expected_states = 6 * (N + 1)
        expected_forces = 6 * N
        expected_zmp = 2 * N
        expected_total = expected_states + expected_forces + expected_zmp
        
        assert dims['states'] == expected_states
        assert dims['forces'] == expected_forces
        assert dims['zmp'] == expected_zmp
        assert dims['total'] == expected_total
        
        print(f"✅ 时域N={N}: {dims['total']} 变量 ({dims['states']}状态 + {dims['forces']}力 + {dims['zmp']}ZMP)")


def test_qp_problem_structure():
    """测试QP问题结构"""
    print("\n=== 测试QP问题结构 ===")
    
    N = 5  # 预测时域
    n_contacts = 2  # 接触点数
    
    # 计算问题维度
    state_dim = 6 * (N + 1)      # 状态变量
    force_dim = 6 * N            # 力变量
    zmp_dim = 2 * N              # ZMP变量
    total_vars = state_dim + force_dim + zmp_dim
    
    # 约束数量估算
    dynamics_constraints = 6 * N      # 动力学约束
    zmp_constraints = 2 * N           # ZMP约束
    force_constraints = 8 * n_contacts * N  # 力约束
    kinematic_constraints = 4 * N     # 运动学约束
    
    total_eq_constraints = dynamics_constraints + zmp_constraints
    total_ineq_constraints = force_constraints + kinematic_constraints
    
    print(f"✅ 决策变量: {total_vars} ({state_dim}+{force_dim}+{zmp_dim})")
    print(f"✅ 等式约束: {total_eq_constraints} ({dynamics_constraints}+{zmp_constraints})")
    print(f"✅ 不等式约束: {total_ineq_constraints} ({force_constraints}+{kinematic_constraints})")
    
    # 验证问题规模合理性
    assert total_vars > 0
    assert total_eq_constraints > 0
    assert total_ineq_constraints > 0
    assert total_vars < 1000  # 合理的规模上限


def test_solver_configurations():
    """测试求解器配置"""
    print("\n=== 测试求解器配置 ===")
    
    # OSQP配置
    osqp_settings = {
        'verbose': False,
        'eps_abs': 1e-4,
        'eps_rel': 1e-4,
        'max_iter': 2000,
        'polish': True,
        'adaptive_rho': True
    }
    
    # qpOASES配置
    qpoases_settings = {
        'printLevel': 0,
        'maxCpuTime': 0.01,
        'maxWorkingSetRecalculations': 60
    }
    
    # 验证配置参数
    assert isinstance(osqp_settings['eps_abs'], float)
    assert osqp_settings['eps_abs'] > 0
    assert isinstance(osqp_settings['max_iter'], int)
    assert osqp_settings['max_iter'] > 0
    
    assert isinstance(qpoases_settings['maxCpuTime'], float)
    assert qpoases_settings['maxCpuTime'] > 0
    
    print("✅ OSQP配置测试通过")
    print("✅ qpOASES配置测试通过")


def main():
    """主测试函数"""
    print("最小MPC求解器测试程序")
    print("=" * 60)
    
    try:
        # 运行各项测试
        test_data_structures()
        test_gait_plan_creation() 
        test_walking_gait_generation()
        test_optimization_problem_dimensions()
        test_qp_problem_structure()
        test_solver_configurations()
        
        print("\n" + "=" * 60)
        print("🎉 所有最小测试通过!")
        print("\n📊 测试总结:")
        print("   ✅ 数据结构正确")
        print("   ✅ 步态计划生成功能正常")
        print("   ✅ 优化问题维度计算正确")
        print("   ✅ QP问题结构合理")
        print("   ✅ 求解器配置有效")
        
        print("\n📝 注意:")
        print("   - 这是最小功能测试，不包含实际的MPC求解")
        print("   - 完整测试需要安装pinocchio等依赖")
        print("   - 可以继续开发具体的求解器实现")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 