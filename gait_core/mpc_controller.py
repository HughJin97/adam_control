"""
基于LIPM的模型预测控制器 (MPC Controller)
实现质心轨迹跟踪和步态规划的优化控制

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("警告: cvxpy未安装，将使用scipy.optimize作为后备求解器")

from .simplified_dynamics import SimplifiedDynamicsModel, LIPMState, Vector3D
from .data_bus import DataBus, get_data_bus


class MPCMode(Enum):
    """MPC控制模式"""
    FORCE_CONTROL = 0       # 仅控制足底力
    FOOTSTEP_CONTROL = 1    # 仅控制落脚点
    COMBINED_CONTROL = 2    # 同时控制力和落脚点


@dataclass
class MPCParameters:
    """MPC参数配置"""
    # 时域参数
    prediction_horizon: int = 20        # 预测时域长度N
    control_horizon: int = 10           # 控制时域长度
    dt: float = 0.1                     # 离散时间步长 [s]
    
    # 代价函数权重
    Q_position: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0]))  # 位置跟踪权重
    Q_velocity: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))    # 速度跟踪权重
    R_force: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))         # 力控制权重
    R_footstep: np.ndarray = field(default_factory=lambda: np.diag([50.0, 50.0]))    # 落脚点权重
    Q_terminal: np.ndarray = field(default_factory=lambda: np.diag([200.0, 200.0]))  # 终端状态权重
    
    # 平滑性权重
    R_force_rate: float = 5.0           # 力变化率权重
    R_footstep_rate: float = 10.0       # 落脚点变化率权重
    
    # 约束参数
    max_force: float = 200.0            # 最大足底力 [N]
    friction_coefficient: float = 0.7   # 摩擦系数
    max_step_length: float = 0.3        # 最大步长 [m]
    max_step_width: float = 0.2         # 最大步宽 [m]
    
    # 质心约束
    com_position_bounds: Tuple[float, float] = (-1.0, 1.0)  # 质心位置约束 [m]
    com_velocity_bounds: Tuple[float, float] = (-2.0, 2.0)  # 质心速度约束 [m/s]
    
    # 支撑多边形参数
    support_polygon_margin: float = 0.05  # 支撑多边形边界 [m]


@dataclass
class MPCReference:
    """MPC参考轨迹"""
    # 参考质心轨迹
    com_position_ref: List[Vector3D] = field(default_factory=list)
    com_velocity_ref: List[Vector3D] = field(default_factory=list)
    
    # 参考落脚点序列
    footstep_ref: List[Vector3D] = field(default_factory=list)
    footstep_timing: List[float] = field(default_factory=list)  # 落脚时间
    
    # 支撑腿序列 ('left' 或 'right')
    support_leg_sequence: List[str] = field(default_factory=list)


@dataclass
class MPCSolution:
    """MPC求解结果"""
    # 优化结果
    optimal_forces: List[Vector3D] = field(default_factory=list)
    optimal_footsteps: List[Vector3D] = field(default_factory=list)
    predicted_trajectory: List[LIPMState] = field(default_factory=list)
    
    # 求解信息
    solve_time: float = 0.0
    cost: float = 0.0
    success: bool = False
    solver_status: str = ""
    
    # 下一步控制输入
    next_force_command: Vector3D = field(default_factory=Vector3D)
    next_footstep_command: Vector3D = field(default_factory=Vector3D)


class MPCController:
    """
    基于LIPM的模型预测控制器
    
    功能：
    1. 基于LIPM模型进行预测
    2. 优化质心轨迹跟踪
    3. 同时优化足底力和落脚点
    4. 满足物理约束和稳定性要求
    """
    
    def __init__(self, 
                 lipm_model: SimplifiedDynamicsModel,
                 mpc_params: Optional[MPCParameters] = None,
                 data_bus: Optional[DataBus] = None):
        """
        初始化MPC控制器
        
        Args:
            lipm_model: LIPM动力学模型
            mpc_params: MPC参数配置
            data_bus: 数据总线
        """
        self.lipm_model = lipm_model
        self.params = mpc_params if mpc_params is not None else MPCParameters()
        self.data_bus = data_bus if data_bus is not None else get_data_bus()
        
        # 控制模式
        self.control_mode = MPCMode.COMBINED_CONTROL
        
        # 参考轨迹
        self.reference = MPCReference()
        
        # 上一步的解
        self.previous_solution = MPCSolution()
        
        # 优化器选择
        self.use_cvxpy = CVXPY_AVAILABLE
        
        # 性能统计
        self.solve_times = []
        self.costs = []
        
        print(f"MPC控制器初始化完成")
        print(f"预测时域: {self.params.prediction_horizon}")
        print(f"控制时域: {self.params.control_horizon}")
        print(f"时间步长: {self.params.dt:.3f}s")
        print(f"求解器: {'CVXPY' if self.use_cvxpy else 'SciPy'}")
    
    def set_reference_trajectory(self, reference: MPCReference):
        """设置参考轨迹"""
        self.reference = reference
        print(f"设置参考轨迹: {len(reference.com_position_ref)} 个位置点")
    
    def generate_walking_reference(self, 
                                   target_velocity: Vector3D,
                                   step_length: float = 0.15,
                                   step_width: float = 0.1,
                                   step_duration: float = 0.6) -> MPCReference:
        """
        生成行走参考轨迹
        
        Args:
            target_velocity: 目标行走速度
            step_length: 步长
            step_width: 步宽
            step_duration: 每步时间
            
        Returns:
            MPCReference: 生成的参考轨迹
        """
        reference = MPCReference()
        
        # 时间序列
        time_horizon = self.params.prediction_horizon * self.params.dt
        times = np.arange(0, time_horizon, self.params.dt)
        
        # 生成质心参考轨迹
        current_com_pos = self.lipm_model.current_state.com_position
        
        for t in times:
            # 简单的直线运动参考
            ref_pos = Vector3D(
                x=current_com_pos.x + target_velocity.x * t,
                y=current_com_pos.y + target_velocity.y * t,
                z=current_com_pos.z
            )
            ref_vel = Vector3D(
                x=target_velocity.x,
                y=target_velocity.y,
                z=0.0
            )
            
            reference.com_position_ref.append(ref_pos)
            reference.com_velocity_ref.append(ref_vel)
        
        # 生成落脚点参考序列
        steps_in_horizon = int(time_horizon / step_duration) + 1
        current_foot_x = 0.0
        
        for i in range(steps_in_horizon):
            # 交替左右脚
            if i % 2 == 0:
                foot_y = step_width / 2  # 左脚
                support_leg = 'left'
            else:
                foot_y = -step_width / 2  # 右脚
                support_leg = 'right'
            
            foot_x = current_foot_x + i * step_length
            
            footstep = Vector3D(x=foot_x, y=foot_y, z=0.0)
            timing = i * step_duration
            
            reference.footstep_ref.append(footstep)
            reference.footstep_timing.append(timing)
            reference.support_leg_sequence.append(support_leg)
        
        return reference
    
    def solve_mpc(self, current_state: LIPMState) -> MPCSolution:
        """
        求解MPC优化问题
        
        Args:
            current_state: 当前状态
            
        Returns:
            MPCSolution: 优化解
        """
        start_time = time.time()
        
        try:
            if self.use_cvxpy and CVXPY_AVAILABLE:
                solution = self._solve_with_cvxpy(current_state)
            else:
                solution = self._solve_with_scipy(current_state)
                
        except Exception as e:
            print(f"MPC求解失败: {e}")
            solution = self._get_fallback_solution(current_state)
        
        solution.solve_time = time.time() - start_time
        self.solve_times.append(solution.solve_time)
        self.costs.append(solution.cost)
        
        self.previous_solution = solution
        return solution
    
    def _solve_with_cvxpy(self, current_state: LIPMState) -> MPCSolution:
        """使用CVXPY求解器"""
        N = self.params.prediction_horizon
        dt = self.params.dt
        
        # 决策变量
        # 状态变量: [x, y, vx, vy] for each time step
        states = cp.Variable((4, N + 1))
        
        # 控制变量
        if self.control_mode in [MPCMode.FORCE_CONTROL, MPCMode.COMBINED_CONTROL]:
            forces = cp.Variable((2, self.params.control_horizon))  # [fx, fy]
        else:
            forces = None
            
        if self.control_mode in [MPCMode.FOOTSTEP_CONTROL, MPCMode.COMBINED_CONTROL]:
            footsteps = cp.Variable((2, len(self.reference.footstep_ref)))  # [foot_x, foot_y]
        else:
            footsteps = None
        
        # 约束列表
        constraints = []
        
        # 初始状态约束
        constraints.append(states[0, 0] == current_state.com_position.x)
        constraints.append(states[1, 0] == current_state.com_position.y)
        constraints.append(states[2, 0] == current_state.com_velocity.x)
        constraints.append(states[3, 0] == current_state.com_velocity.y)
        
        # 动力学约束 (LIPM模型离散化)
        g = self.lipm_model.parameters.gravity
        z_c = self.lipm_model.parameters.com_height
        omega_n = np.sqrt(g / z_c)
        mass = self.lipm_model.parameters.total_mass
        
        # 离散化系数
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [omega_n**2 * dt, 0, 1, 0],
            [0, omega_n**2 * dt, 0, 1]
        ])
        
        B_force = np.array([
            [0, 0],
            [0, 0],
            [dt/mass, 0],
            [0, dt/mass]
        ]) if forces is not None else np.zeros((4, 2))
        
        # 支撑足位置影响矩阵
        B_foot = np.array([
            [0, 0],
            [0, 0],
            [-omega_n**2 * dt, 0],
            [0, -omega_n**2 * dt]
        ])
        
        for k in range(N):
            # 获取当前时刻的支撑足位置
            if k < len(self.reference.footstep_ref):
                foot_pos = self.reference.footstep_ref[k]
                foot_input = np.array([foot_pos.x, foot_pos.y])
            else:
                foot_input = np.array([0.0, 0.0])
            
            if forces is not None and k < self.params.control_horizon:
                # 有力控制的动力学
                constraints.append(
                    states[:, k+1] == A @ states[:, k] + 
                    B_force @ forces[:, k] + B_foot @ foot_input
                )
            else:
                # 仅步态控制的动力学
                constraints.append(
                    states[:, k+1] == A @ states[:, k] + B_foot @ foot_input
                )
        
        # 状态约束
        pos_bounds = self.params.com_position_bounds
        vel_bounds = self.params.com_velocity_bounds
        
        constraints.append(states[0, :] >= pos_bounds[0])  # x position
        constraints.append(states[0, :] <= pos_bounds[1])
        constraints.append(states[1, :] >= pos_bounds[0])  # y position
        constraints.append(states[1, :] <= pos_bounds[1])
        constraints.append(states[2, :] >= vel_bounds[0])  # x velocity
        constraints.append(states[2, :] <= vel_bounds[1])
        constraints.append(states[3, :] >= vel_bounds[0])  # y velocity
        constraints.append(states[3, :] <= vel_bounds[1])
        
        # 控制约束
        if forces is not None:
            # 力大小约束
            for k in range(self.params.control_horizon):
                constraints.append(cp.norm(forces[:, k]) <= self.params.max_force)
            
            # 摩擦锥约束 (简化为圆形)
            friction_limit = self.params.friction_coefficient * mass * g / 2
            for k in range(self.params.control_horizon):
                constraints.append(cp.norm(forces[:, k]) <= friction_limit)
        
        if footsteps is not None:
            # 落脚点范围约束
            max_step = self.params.max_step_length
            max_width = self.params.max_step_width
            
            for i in range(len(self.reference.footstep_ref)):
                constraints.append(footsteps[0, i] >= -max_step)
                constraints.append(footsteps[0, i] <= max_step)
                constraints.append(footsteps[1, i] >= -max_width)
                constraints.append(footsteps[1, i] <= max_width)
        
        # 构造代价函数
        cost = 0
        
        # 轨迹跟踪代价
        for k in range(N + 1):
            if k < len(self.reference.com_position_ref):
                ref_pos = self.reference.com_position_ref[k]
                pos_error = states[:2, k] - np.array([ref_pos.x, ref_pos.y])
                cost += cp.quad_form(pos_error, self.params.Q_position)
            
            if k < len(self.reference.com_velocity_ref):
                ref_vel = self.reference.com_velocity_ref[k]
                vel_error = states[2:, k] - np.array([ref_vel.x, ref_vel.y])
                cost += cp.quad_form(vel_error, self.params.Q_velocity)
        
        # 终端代价
        if len(self.reference.com_position_ref) > 0:
            ref_pos = self.reference.com_position_ref[-1]
            terminal_error = states[:2, -1] - np.array([ref_pos.x, ref_pos.y])
            cost += cp.quad_form(terminal_error, self.params.Q_terminal)
        
        # 控制代价
        if forces is not None:
            for k in range(self.params.control_horizon):
                cost += cp.quad_form(forces[:, k], self.params.R_force)
            
            # 控制平滑性
            for k in range(self.params.control_horizon - 1):
                force_diff = forces[:, k+1] - forces[:, k]
                cost += self.params.R_force_rate * cp.sum_squares(force_diff)
        
        if footsteps is not None:
            for i in range(len(self.reference.footstep_ref)):
                if i < len(self.reference.footstep_ref):
                    ref_foot = self.reference.footstep_ref[i]
                    foot_error = footsteps[:, i] - np.array([ref_foot.x, ref_foot.y])
                    cost += cp.quad_form(foot_error, self.params.R_footstep)
        
        # 求解优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        # 提取解
        solution = MPCSolution()
        solution.success = problem.status == cp.OPTIMAL
        solution.solver_status = problem.status
        solution.cost = problem.value if problem.value is not None else float('inf')
        
        if solution.success:
            # 提取优化结果
            if forces is not None:
                for k in range(self.params.control_horizon):
                    force = Vector3D(x=forces[0, k].value, y=forces[1, k].value, z=0.0)
                    solution.optimal_forces.append(force)
                
                # 下一步控制命令
                solution.next_force_command = solution.optimal_forces[0]
            
            if footsteps is not None:
                for i in range(len(self.reference.footstep_ref)):
                    foot = Vector3D(x=footsteps[0, i].value, y=footsteps[1, i].value, z=0.0)
                    solution.optimal_footsteps.append(foot)
                
                if solution.optimal_footsteps:
                    solution.next_footstep_command = solution.optimal_footsteps[0]
            
            # 预测轨迹
            for k in range(N + 1):
                state = LIPMState()
                state.com_position = Vector3D(
                    x=states[0, k].value,
                    y=states[1, k].value,
                    z=current_state.com_position.z
                )
                state.com_velocity = Vector3D(
                    x=states[2, k].value,
                    y=states[3, k].value,
                    z=0.0
                )
                state.timestamp = current_state.timestamp + k * dt
                solution.predicted_trajectory.append(state)
        
        return solution
    
    def _solve_with_scipy(self, current_state: LIPMState) -> MPCSolution:
        """使用SciPy求解器（简化版本）"""
        N = self.params.prediction_horizon
        dt = self.params.dt
        
        # 决策变量维度
        n_forces = 2 * self.params.control_horizon if self.control_mode != MPCMode.FOOTSTEP_CONTROL else 0
        n_footsteps = 2 * len(self.reference.footstep_ref) if self.control_mode != MPCMode.FORCE_CONTROL else 0
        n_vars = n_forces + n_footsteps
        
        if n_vars == 0:
            return self._get_fallback_solution(current_state)
        
        # 初始猜测
        x0 = np.zeros(n_vars)
        
        # 边界约束
        bounds = []
        
        # 力的边界
        for i in range(n_forces // 2):
            bounds.append((-self.params.max_force, self.params.max_force))  # fx
            bounds.append((-self.params.max_force, self.params.max_force))  # fy
        
        # 落脚点的边界
        for i in range(n_footsteps // 2):
            bounds.append((-self.params.max_step_length, self.params.max_step_length))  # foot_x
            bounds.append((-self.params.max_step_width, self.params.max_step_width))    # foot_y
        
        # 优化求解
        result = opt.minimize(
            fun=lambda x: self._evaluate_cost_scipy(x, current_state),
            x0=x0,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 100, 'disp': False}
        )
        
        # 构造解
        solution = MPCSolution()
        solution.success = result.success
        solution.solver_status = "SLSQP: " + ("Success" if result.success else "Failed")
        solution.cost = result.fun if result.success else float('inf')
        
        if result.success:
            # 提取力控制结果
            if n_forces > 0:
                forces_flat = result.x[:n_forces]
                for i in range(0, n_forces, 2):
                    force = Vector3D(x=forces_flat[i], y=forces_flat[i+1], z=0.0)
                    solution.optimal_forces.append(force)
                
                if solution.optimal_forces:
                    solution.next_force_command = solution.optimal_forces[0]
            
            # 提取落脚点结果
            if n_footsteps > 0:
                footsteps_flat = result.x[n_forces:n_forces+n_footsteps]
                for i in range(0, n_footsteps, 2):
                    foot = Vector3D(x=footsteps_flat[i], y=footsteps_flat[i+1], z=0.0)
                    solution.optimal_footsteps.append(foot)
                
                if solution.optimal_footsteps:
                    solution.next_footstep_command = solution.optimal_footsteps[0]
            
            # 生成预测轨迹
            solution.predicted_trajectory = self._simulate_trajectory(
                current_state, solution.optimal_forces, solution.optimal_footsteps
            )
        
        return solution
    
    def _evaluate_cost_scipy(self, x: np.ndarray, current_state: LIPMState) -> float:
        """评估SciPy优化的代价函数"""
        try:
            # 解析决策变量
            n_forces = 2 * self.params.control_horizon if self.control_mode != MPCMode.FOOTSTEP_CONTROL else 0
            
            optimal_forces = []
            optimal_footsteps = []
            
            if n_forces > 0:
                forces_flat = x[:n_forces]
                for i in range(0, n_forces, 2):
                    force = Vector3D(x=forces_flat[i], y=forces_flat[i+1], z=0.0)
                    optimal_forces.append(force)
            
            if len(x) > n_forces:
                footsteps_flat = x[n_forces:]
                for i in range(0, len(footsteps_flat), 2):
                    foot = Vector3D(x=footsteps_flat[i], y=footsteps_flat[i+1], z=0.0)
                    optimal_footsteps.append(foot)
            
            # 模拟轨迹
            trajectory = self._simulate_trajectory(current_state, optimal_forces, optimal_footsteps)
            
            # 计算代价
            cost = 0.0
            
            # 轨迹跟踪代价
            for k, state in enumerate(trajectory):
                if k < len(self.reference.com_position_ref):
                    ref_pos = self.reference.com_position_ref[k]
                    pos_error = np.array([
                        state.com_position.x - ref_pos.x,
                        state.com_position.y - ref_pos.y
                    ])
                    cost += pos_error.T @ self.params.Q_position @ pos_error
                
                if k < len(self.reference.com_velocity_ref):
                    ref_vel = self.reference.com_velocity_ref[k]
                    vel_error = np.array([
                        state.com_velocity.x - ref_vel.x,
                        state.com_velocity.y - ref_vel.y
                    ])
                    cost += vel_error.T @ self.params.Q_velocity @ vel_error
            
            # 控制代价
            for force in optimal_forces:
                force_vec = np.array([force.x, force.y])
                cost += force_vec.T @ self.params.R_force @ force_vec
            
            for i, foot in enumerate(optimal_footsteps):
                if i < len(self.reference.footstep_ref):
                    ref_foot = self.reference.footstep_ref[i]
                    foot_error = np.array([
                        foot.x - ref_foot.x,
                        foot.y - ref_foot.y
                    ])
                    cost += foot_error.T @ self.params.R_footstep @ foot_error
            
            return cost
            
        except Exception as e:
            print(f"代价函数评估失败: {e}")
            return float('inf')
    
    def _simulate_trajectory(self, 
                           initial_state: LIPMState,
                           forces: List[Vector3D],
                           footsteps: List[Vector3D]) -> List[LIPMState]:
        """模拟预测轨迹"""
        trajectory = [initial_state]
        current_state = initial_state
        
        for k in range(self.params.prediction_horizon):
            # 设置控制输入
            if k < len(forces):
                self.lipm_model.set_foot_force_command(forces[k])
            
            if k < len(footsteps):
                self.lipm_model.set_support_foot_position(footsteps[k])
            elif k < len(self.reference.footstep_ref):
                self.lipm_model.set_support_foot_position(self.reference.footstep_ref[k])
            
            # 模拟一步
            self.lipm_model.current_state = current_state
            next_state = self.lipm_model.compute_com_dynamics(self.params.dt)
            
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def _get_fallback_solution(self, current_state: LIPMState) -> MPCSolution:
        """获取后备解决方案"""
        solution = MPCSolution()
        solution.success = False
        solution.solver_status = "Fallback"
        solution.cost = float('inf')
        
        # 使用零控制输入
        solution.next_force_command = Vector3D(x=0.0, y=0.0, z=0.0)
        if self.reference.footstep_ref:
            solution.next_footstep_command = self.reference.footstep_ref[0]
        else:
            solution.next_footstep_command = Vector3D(x=0.0, y=0.0, z=0.0)
        
        return solution
    
    def update_control(self, dt: float) -> MPCSolution:
        """
        MPC控制更新
        
        Args:
            dt: 时间步长
            
        Returns:
            MPCSolution: MPC解
        """
        # 从数据总线获取当前状态
        self.lipm_model._update_state_from_data_bus()
        current_state = self.lipm_model.current_state
        
        # 求解MPC
        solution = self.solve_mpc(current_state)
        
        if solution.success:
            # 应用控制命令
            if solution.next_force_command:
                self.lipm_model.set_foot_force_command(solution.next_force_command)
            
            if solution.next_footstep_command:
                self.lipm_model.set_support_foot_position(solution.next_footstep_command)
            
            # 更新LIPM模型
            self.lipm_model.update(dt)
            
            print(f"MPC更新成功: 求解时间 {solution.solve_time:.3f}s, 代价 {solution.cost:.2f}")
        else:
            print(f"MPC求解失败: {solution.solver_status}")
        
        return solution
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.solve_times:
            return {}
        
        return {
            'average_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'average_cost': np.mean(self.costs),
            'total_solves': len(self.solve_times)
        }
    
    def print_status(self):
        """打印控制器状态"""
        print("\n=== MPC控制器状态 ===")
        print(f"控制模式: {self.control_mode.name}")
        print(f"预测时域: {self.params.prediction_horizon}")
        print(f"控制时域: {self.params.control_horizon}")
        print(f"时间步长: {self.params.dt:.3f}s")
        print(f"参考轨迹长度: {len(self.reference.com_position_ref)}")
        
        stats = self.get_performance_stats()
        if stats:
            print(f"平均求解时间: {stats['average_solve_time']:.3f}s")
            print(f"平均代价: {stats['average_cost']:.2f}")
            print(f"求解次数: {stats['total_solves']}")
        
        print("====================\n")


def create_mpc_controller(lipm_model: SimplifiedDynamicsModel,
                         prediction_horizon: int = 20,
                         control_horizon: int = 10,
                         dt: float = 0.1) -> MPCController:
    """
    创建MPC控制器实例
    
    Args:
        lipm_model: LIPM动力学模型
        prediction_horizon: 预测时域长度
        control_horizon: 控制时域长度
        dt: 时间步长
        
    Returns:
        MPCController: MPC控制器实例
    """
    params = MPCParameters()
    params.prediction_horizon = prediction_horizon
    params.control_horizon = control_horizon
    params.dt = dt
    
    return MPCController(lipm_model, params) 