"""
专业MPC求解器实现
集成OSQP、qpOASES等QP求解器，实现高效的机器人控制优化

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# QP求解器导入
try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

try:
    import qpoases
    QPOASES_AVAILABLE = True
except ImportError:
    QPOASES_AVAILABLE = False

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from .simplified_dynamics import SimplifiedDynamicsModel, LIPMState, Vector3D
from .data_bus import DataBus, get_data_bus


class QPSolver(Enum):
    """QP求解器类型"""
    OSQP = "osqp"           # 推荐：快速、鲁棒
    QPOASES = "qpoases"     # 高精度
    CVXPY_OSQP = "cvxpy_osqp"  # CVXPY包装的OSQP
    SCIPY = "scipy"         # 后备方案


@dataclass
class GaitPlan:
    """步态计划数据结构"""
    # 支撑腿序列（每个时间步的支撑状态）
    support_sequence: List[str] = field(default_factory=list)  # ['left', 'right', 'double', 'flight']
    
    # 接触状态矩阵 [time_steps x n_contacts]
    contact_schedule: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 支撑足位置轨迹
    support_positions: List[Dict[str, Vector3D]] = field(default_factory=list)
    
    # 步态时间信息
    step_times: List[float] = field(default_factory=list)
    phase_durations: List[float] = field(default_factory=list)
    
    # 步态约束
    min_contact_duration: float = 0.1  # 最小接触时间
    max_step_length: float = 0.3       # 最大步长
    step_height: float = 0.05          # 步高


@dataclass 
class MPCResult:
    """MPC求解结果"""
    # 优化状态
    success: bool = False
    solve_time: float = 0.0
    cost: float = float('inf')
    solver_info: Dict = field(default_factory=dict)
    
    # 支撑力轨迹 [time_steps x n_contacts x 3]
    contact_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 质心轨迹
    com_position_trajectory: List[Vector3D] = field(default_factory=list)
    com_velocity_trajectory: List[Vector3D] = field(default_factory=list)
    com_acceleration_trajectory: List[Vector3D] = field(default_factory=list)
    
    # ZMP轨迹
    zmp_trajectory: List[Vector3D] = field(default_factory=list)
    
    # 下一步最优落脚位置
    next_footstep: Dict[str, Vector3D] = field(default_factory=dict)
    
    # 优化轨迹（完整预测）
    predicted_com_states: List[LIPMState] = field(default_factory=list)
    
    # 控制命令（当前时刻）
    current_contact_forces: Dict[str, Vector3D] = field(default_factory=dict)
    current_desired_com_acceleration: Vector3D = field(default_factory=Vector3D)


class MPCSolver:
    """
    专业MPC求解器
    
    参考OpenLoong框架的设计理念：
    - 输入：当前状态 + 步态计划
    - 输出：支撑力轨迹 + 质心轨迹 + 落脚点
    """
    
    def __init__(self, 
                 lipm_model: SimplifiedDynamicsModel,
                 solver_type: QPSolver = QPSolver.OSQP,
                 prediction_horizon: int = 20,
                 control_horizon: int = 10,
                 dt: float = 0.1):
        """
        初始化MPC求解器
        
        Args:
            lipm_model: LIPM动力学模型
            solver_type: QP求解器类型
            prediction_horizon: 预测时域
            control_horizon: 控制时域  
            dt: 时间步长
        """
        self.lipm_model = lipm_model
        self.solver_type = solver_type
        self.N = prediction_horizon
        self.M = control_horizon
        self.dt = dt
        
        # 系统参数
        self.g = lipm_model.parameters.gravity
        self.z_c = lipm_model.parameters.com_height
        self.mass = lipm_model.parameters.total_mass
        self.omega = np.sqrt(self.g / self.z_c)
        
        # 优化参数
        self.setup_optimization_parameters()
        
        # 求解器设置
        self.setup_solver()
        
        # 性能统计
        self.solve_times = []
        self.success_rate = []
        
        print(f"MPC求解器初始化: {solver_type.value}, 时域={prediction_horizon}, dt={dt:.3f}s")
    
    def setup_optimization_parameters(self):
        """设置优化参数"""
        # 状态权重
        self.Q_pos = np.diag([100.0, 100.0])      # 位置跟踪
        self.Q_vel = np.diag([10.0, 10.0])        # 速度跟踪
        self.Q_acc = np.diag([1.0, 1.0])          # 加速度平滑
        self.Q_terminal = np.diag([200.0, 200.0]) # 终端状态
        
        # 控制权重
        self.R_force = np.diag([0.1, 0.1, 0.01])  # [fx, fy, fz]
        self.R_zmp = np.diag([50.0, 50.0])        # ZMP跟踪
        
        # 平滑性权重
        self.W_force_rate = 5.0    # 力变化率
        self.W_jerk = 2.0          # 加加速度
        
        # 约束参数
        self.f_max = 200.0         # 最大接触力
        self.mu = 0.7              # 摩擦系数
        self.zmp_margin = 0.02     # ZMP裕量
        
        # 接触点数量
        self.n_contacts = 2        # 双足机器人
        
    def setup_solver(self):
        """设置QP求解器"""
        if self.solver_type == QPSolver.OSQP and OSQP_AVAILABLE:
            self.solver_settings = {
                'verbose': False,
                'eps_abs': 1e-4,
                'eps_rel': 1e-4,
                'max_iter': 2000,
                'polish': True,
                'adaptive_rho': True
            }
            print("使用OSQP求解器")
            
        elif self.solver_type == QPSolver.QPOASES and QPOASES_AVAILABLE:
            self.solver_settings = {
                'printLevel': 0,
                'maxCpuTime': 0.01,  # 10ms限制
                'maxWorkingSetRecalculations': 60
            }
            print("使用qpOASES求解器")
            
        else:
            self.solver_type = QPSolver.SCIPY
            print("使用SciPy后备求解器")
    
    def solve(self, current_state: LIPMState, gait_plan: GaitPlan) -> MPCResult:
        """
        MPC主求解函数
        
        Args:
            current_state: 当前机器人状态（质心位置、速度等）
            gait_plan: 步态计划（支撑序列、接触时间等）
            
        Returns:
            MPCResult: 优化结果（力轨迹、质心轨迹、落脚点等）
        """
        start_time = time.time()
        
        try:
            # 1. 构建QP问题
            H, f, A_eq, b_eq, A_ineq, b_ineq = self._formulate_qp_problem(
                current_state, gait_plan
            )
            
            # 2. 求解QP
            solution = self._solve_qp(H, f, A_eq, b_eq, A_ineq, b_ineq)
            
            # 3. 解析结果
            result = self._parse_solution(solution, current_state, gait_plan)
            
            result.solve_time = time.time() - start_time
            self.solve_times.append(result.solve_time)
            self.success_rate.append(1.0 if result.success else 0.0)
            
            if result.success:
                print(f"MPC求解成功: 代价={result.cost:.3f}, 时间={result.solve_time:.3f}s")
            else:
                print(f"MPC求解失败: {result.solver_info}")
                
            return result
            
        except Exception as e:
            print(f"MPC求解异常: {e}")
            return self._get_fallback_result(current_state)
    
    def _formulate_qp_problem(self, current_state: LIPMState, gait_plan: GaitPlan) -> Tuple:
        """构建QP问题"""
        N, M = self.N, self.M
        n_vars = self._get_variable_dimensions(gait_plan)
        
        # 决策变量: [com_states, contact_forces, zmp_positions]
        # com_states: [x, y, vx, vy, ax, ay] * (N+1)
        # contact_forces: [fx_left, fy_left, fz_left, fx_right, fy_right, fz_right] * N
        # zmp_positions: [zmp_x, zmp_y] * N
        
        state_dim = 6 * (N + 1)      # 位置、速度、加速度
        force_dim = 6 * N            # 双足的3D力
        zmp_dim = 2 * N              # ZMP位置
        
        total_vars = state_dim + force_dim + zmp_dim
        
        # 1. 构建Hessian矩阵H (代价函数二次项)
        H = sp.csc_matrix((total_vars, total_vars))
        H = self._build_hessian_matrix(H, gait_plan)
        
        # 2. 构建线性项f
        f = np.zeros(total_vars)
        f = self._build_linear_cost(f, current_state, gait_plan)
        
        # 3. 构建等式约束 (动力学约束)
        A_eq, b_eq = self._build_dynamics_constraints(current_state, gait_plan)
        
        # 4. 构建不等式约束 (物理约束)
        A_ineq, b_ineq = self._build_inequality_constraints(gait_plan)
        
        return H, f, A_eq, b_eq, A_ineq, b_ineq
    
    def _get_variable_dimensions(self, gait_plan: GaitPlan) -> Dict:
        """获取决策变量维度"""
        N = self.N
        return {
            'states': 6 * (N + 1),     # [x,y,vx,vy,ax,ay] * (N+1)
            'forces': 6 * N,           # [fx,fy,fz] * 2_feet * N  
            'zmp': 2 * N,              # [zmp_x, zmp_y] * N
            'total': 6 * (N + 1) + 6 * N + 2 * N
        }
    
    def _build_hessian_matrix(self, H: sp.csc_matrix, gait_plan: GaitPlan) -> sp.csc_matrix:
        """构建Hessian矩阵（代价函数二次项系数）"""
        N = self.N
        dims = self._get_variable_dimensions(gait_plan)
        
        # 状态跟踪代价权重
        for k in range(N + 1):
            # 位置权重 [x, y]
            pos_idx = k * 6
            H[pos_idx:pos_idx+2, pos_idx:pos_idx+2] = self.Q_pos
            
            # 速度权重 [vx, vy]  
            vel_idx = k * 6 + 2
            H[vel_idx:vel_idx+2, vel_idx:vel_idx+2] = self.Q_vel
            
            # 加速度权重 [ax, ay]
            acc_idx = k * 6 + 4
            H[acc_idx:acc_idx+2, acc_idx:acc_idx+2] = self.Q_acc
        
        # 终端状态权重
        terminal_pos_idx = N * 6
        H[terminal_pos_idx:terminal_pos_idx+2, terminal_pos_idx:terminal_pos_idx+2] = self.Q_terminal
        
        # 控制输入权重（接触力）
        force_start_idx = dims['states']
        for k in range(N):
            # 左脚力权重
            left_force_idx = force_start_idx + k * 6
            H[left_force_idx:left_force_idx+3, left_force_idx:left_force_idx+3] = self.R_force
            
            # 右脚力权重  
            right_force_idx = force_start_idx + k * 6 + 3
            H[right_force_idx:right_force_idx+3, right_force_idx:right_force_idx+3] = self.R_force
        
        # ZMP跟踪权重
        zmp_start_idx = dims['states'] + dims['forces']
        for k in range(N):
            zmp_idx = zmp_start_idx + k * 2
            H[zmp_idx:zmp_idx+2, zmp_idx:zmp_idx+2] = self.R_zmp
            
        # 平滑性权重（力变化率）
        for k in range(N - 1):
            curr_force_idx = force_start_idx + k * 6
            next_force_idx = force_start_idx + (k + 1) * 6
            
            # 添加 ||f[k+1] - f[k]||² 项
            for i in range(6):
                H[curr_force_idx + i, curr_force_idx + i] += self.W_force_rate
                H[next_force_idx + i, next_force_idx + i] += self.W_force_rate
                H[curr_force_idx + i, next_force_idx + i] = -self.W_force_rate
                H[next_force_idx + i, curr_force_idx + i] = -self.W_force_rate
        
        return H.tocsc()
    
    def _build_linear_cost(self, f: np.ndarray, current_state: LIPMState, gait_plan: GaitPlan) -> np.ndarray:
        """构建线性代价项"""
        N = self.N
        
        # 参考轨迹生成（这里使用简单的直线运动）
        ref_trajectory = self._generate_reference_trajectory(current_state, gait_plan)
        
        # 状态跟踪线性项: -2 * Q * x_ref
        for k in range(N + 1):
            if k < len(ref_trajectory):
                ref_state = ref_trajectory[k]
                
                # 位置参考
                pos_idx = k * 6
                f[pos_idx] = -2 * self.Q_pos[0, 0] * ref_state.com_position.x
                f[pos_idx + 1] = -2 * self.Q_pos[1, 1] * ref_state.com_position.y
                
                # 速度参考
                vel_idx = k * 6 + 2
                f[vel_idx] = -2 * self.Q_vel[0, 0] * ref_state.com_velocity.x
                f[vel_idx + 1] = -2 * self.Q_vel[1, 1] * ref_state.com_velocity.y
        
        return f
    
    def _generate_reference_trajectory(self, current_state: LIPMState, gait_plan: GaitPlan) -> List[LIPMState]:
        """生成参考轨迹"""
        trajectory = []
        
        # 简单的常速直线运动参考
        current_vel = current_state.com_velocity
        
        for k in range(self.N + 1):
            t = k * self.dt
            ref_state = LIPMState()
            
            # 位置：匀速运动
            ref_state.com_position = Vector3D(
                x=current_state.com_position.x + current_vel.x * t,
                y=current_state.com_position.y + current_vel.y * t,
                z=self.z_c
            )
            
            # 速度：保持恒定
            ref_state.com_velocity = Vector3D(
                x=current_vel.x,
                y=current_vel.y,
                z=0.0
            )
            
            # 加速度：为零（匀速运动）
            ref_state.com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
            ref_state.timestamp = current_state.timestamp + t
            
            trajectory.append(ref_state)
        
        return trajectory
    
    def _build_dynamics_constraints(self, current_state: LIPMState, gait_plan: GaitPlan) -> Tuple[sp.csc_matrix, np.ndarray]:
        """构建动力学等式约束"""
        N = self.N
        dims = self._get_variable_dimensions(gait_plan)
        
        # 约束数量：LIPM动力学 + ZMP约束
        n_dynamics_constraints = 6 * N     # [x, y, vx, vy, ax, ay] for each step
        n_zmp_constraints = 2 * N          # ZMP = f(com_state, forces)
        n_constraints = n_dynamics_constraints + n_zmp_constraints
        
        A_eq = sp.lil_matrix((n_constraints, dims['total']))
        b_eq = np.zeros(n_constraints)
        
        # 1. LIPM动力学约束: x[k+1] = A*x[k] + B*u[k]
        dt = self.dt
        omega2 = self.omega ** 2
        
        # 离散化状态转移矩阵
        A_sys = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],        # x[k+1] = x[k] + vx*dt + 0.5*ax*dt²
            [0, 1, 0, dt, 0, 0.5*dt**2],        # y[k+1] = y[k] + vy*dt + 0.5*ay*dt²
            [0, 0, 1, 0, dt, 0],                # vx[k+1] = vx[k] + ax*dt
            [0, 0, 0, 1, 0, dt],                # vy[k+1] = vy[k] + ay*dt
            [omega2, 0, 0, 0, 0, 0],            # ax[k+1] = ω²*x[k] + force_terms
            [0, omega2, 0, 0, 0, 0]             # ay[k+1] = ω²*y[k] + force_terms
        ])
        
        # 力输入矩阵（接触力对加速度的影响）
        B_force = np.zeros((6, 6))  # [state_dim x force_dim_per_step]
        B_force[4, 0] = 1.0 / self.mass    # ax += fx_left/m
        B_force[4, 3] = 1.0 / self.mass    # ax += fx_right/m  
        B_force[5, 1] = 1.0 / self.mass    # ay += fy_left/m
        B_force[5, 4] = 1.0 / self.mass    # ay += fy_right/m
        
        # 设置初始状态约束
        current_state_vec = np.array([
            current_state.com_position.x,
            current_state.com_position.y,
            current_state.com_velocity.x,
            current_state.com_velocity.y,
            current_state.com_acceleration.x,
            current_state.com_acceleration.y
        ])
        
        constraint_idx = 0
        
        # 动力学约束: x[k+1] - A*x[k] - B*u[k] = 0
        for k in range(N):
            # 状态变量索引
            curr_state_idx = k * 6
            next_state_idx = (k + 1) * 6
            
            # 力变量索引
            force_idx = dims['states'] + k * 6
            
            # x[k+1] 系数
            for i in range(6):
                A_eq[constraint_idx + i, next_state_idx + i] = 1.0
            
            # -A*x[k] 系数
            for i in range(6):
                for j in range(6):
                    if A_sys[i, j] != 0:
                        A_eq[constraint_idx + i, curr_state_idx + j] = -A_sys[i, j]
            
            # -B*u[k] 系数
            for i in range(6):
                for j in range(6):
                    if B_force[i, j] != 0:
                        A_eq[constraint_idx + i, force_idx + j] = -B_force[i, j]
            
            # 支撑足位置的影响（LIMP方程中的-ω²*p_foot项）
            if k < len(gait_plan.support_positions):
                support_pos = gait_plan.support_positions[k]
                if 'left' in support_pos or 'right' in support_pos:
                    # 简化：使用平均支撑位置
                    avg_foot_x = 0.0
                    avg_foot_y = 0.0
                    n_feet = 0
                    
                    if 'left' in support_pos:
                        avg_foot_x += support_pos['left'].x
                        avg_foot_y += support_pos['left'].y
                        n_feet += 1
                    if 'right' in support_pos:
                        avg_foot_x += support_pos['right'].x
                        avg_foot_y += support_pos['right'].y
                        n_feet += 1
                    
                    if n_feet > 0:
                        avg_foot_x /= n_feet
                        avg_foot_y /= n_feet
                        
                        # 添加到约束右端
                        b_eq[constraint_idx + 4] = -omega2 * avg_foot_x  # ax项
                        b_eq[constraint_idx + 5] = -omega2 * avg_foot_y  # ay项
            
            constraint_idx += 6
        
        # 2. ZMP约束: zmp = com_pos - (z_c/g) * com_acc
        zmp_start_idx = dims['states'] + dims['forces']
        for k in range(N):
            state_idx = k * 6
            zmp_idx = zmp_start_idx + k * 2
            
            # zmp_x = x - (z_c/g) * ax
            A_eq[constraint_idx, state_idx] = 1.0      # x系数
            A_eq[constraint_idx, state_idx + 4] = -self.z_c / self.g  # ax系数
            A_eq[constraint_idx, zmp_idx] = -1.0       # zmp_x系数
            
            # zmp_y = y - (z_c/g) * ay  
            A_eq[constraint_idx + 1, state_idx + 1] = 1.0    # y系数
            A_eq[constraint_idx + 1, state_idx + 5] = -self.z_c / self.g  # ay系数
            A_eq[constraint_idx + 1, zmp_idx + 1] = -1.0     # zmp_y系数
            
            constraint_idx += 2
        
        return A_eq.tocsc(), b_eq
    
    def _build_inequality_constraints(self, gait_plan: GaitPlan) -> Tuple[sp.csc_matrix, np.ndarray]:
        """构建不等式约束"""
        N = self.N
        dims = self._get_variable_dimensions(gait_plan)
        
        # 约束类型：
        # 1. 接触力界限: 0 <= fz <= f_max, |fx|,|fy| <= mu*fz
        # 2. ZMP约束: ZMP在支撑多边形内
        # 3. 运动学界限: 速度、加速度限制
        
        force_constraints_per_step = 8 * self.n_contacts  # 每个接触点8个约束
        zmp_constraints_per_step = 4      # ZMP边界约束
        kinematic_constraints_per_step = 4  # 速度、加速度限制
        
        n_constraints_per_step = force_constraints_per_step + zmp_constraints_per_step + kinematic_constraints_per_step
        total_constraints = n_constraints_per_step * N
        
        A_ineq = sp.lil_matrix((total_constraints, dims['total']))
        b_ineq = np.zeros(total_constraints)
        
        constraint_idx = 0
        force_start_idx = dims['states']
        zmp_start_idx = dims['states'] + dims['forces']
        
        for k in range(N):
            # 1. 接触力约束
            for foot_idx in range(self.n_contacts):  # 0=left, 1=right
                force_idx = force_start_idx + k * 6 + foot_idx * 3  # [fx, fy, fz]
                
                # 检查是否有接触
                has_contact = True
                if k < len(gait_plan.contact_schedule):
                    has_contact = gait_plan.contact_schedule[k, foot_idx] > 0.5
                
                if has_contact:
                    # fz >= 0
                    A_ineq[constraint_idx, force_idx + 2] = -1.0
                    b_ineq[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # fz <= f_max
                    A_ineq[constraint_idx, force_idx + 2] = 1.0
                    b_ineq[constraint_idx] = self.f_max
                    constraint_idx += 1
                    
                    # 摩擦锥约束: |fx| <= mu * fz
                    # fx <= mu * fz  =>  fx - mu*fz <= 0
                    A_ineq[constraint_idx, force_idx] = 1.0      # fx
                    A_ineq[constraint_idx, force_idx + 2] = -self.mu  # -mu*fz
                    b_ineq[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # -fx <= mu * fz  =>  -fx - mu*fz <= 0
                    A_ineq[constraint_idx, force_idx] = -1.0     # -fx
                    A_ineq[constraint_idx, force_idx + 2] = -self.mu  # -mu*fz
                    b_ineq[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    # |fy| <= mu * fz (类似处理)
                    A_ineq[constraint_idx, force_idx + 1] = 1.0    # fy
                    A_ineq[constraint_idx, force_idx + 2] = -self.mu  # -mu*fz
                    b_ineq[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                    A_ineq[constraint_idx, force_idx + 1] = -1.0   # -fy
                    A_ineq[constraint_idx, force_idx + 2] = -self.mu  # -mu*fz
                    b_ineq[constraint_idx] = 0.0
                    constraint_idx += 1
                    
                else:
                    # 无接触时：所有力为零
                    for i in range(3):
                        A_ineq[constraint_idx, force_idx + i] = 1.0
                        b_ineq[constraint_idx] = 0.0
                        constraint_idx += 1
                        
                        A_ineq[constraint_idx, force_idx + i] = -1.0
                        b_ineq[constraint_idx] = 0.0
                        constraint_idx += 1
            
            # 2. ZMP约束（简化为矩形支撑区域）
            zmp_idx = zmp_start_idx + k * 2
            zmp_bounds = 0.1  # 支撑区域半径
            
            # zmp_x >= -zmp_bounds
            A_ineq[constraint_idx, zmp_idx] = -1.0
            b_ineq[constraint_idx] = zmp_bounds
            constraint_idx += 1
            
            # zmp_x <= zmp_bounds
            A_ineq[constraint_idx, zmp_idx] = 1.0
            b_ineq[constraint_idx] = zmp_bounds
            constraint_idx += 1
            
            # zmp_y >= -zmp_bounds
            A_ineq[constraint_idx, zmp_idx + 1] = -1.0
            b_ineq[constraint_idx] = zmp_bounds
            constraint_idx += 1
            
            # zmp_y <= zmp_bounds
            A_ineq[constraint_idx, zmp_idx + 1] = 1.0
            b_ineq[constraint_idx] = zmp_bounds
            constraint_idx += 1
            
            # 3. 运动学约束
            state_idx = k * 6
            vel_bound = 2.0    # 速度限制
            acc_bound = 5.0    # 加速度限制
            
            # 速度限制: |vx|, |vy| <= vel_bound
            A_ineq[constraint_idx, state_idx + 2] = 1.0   # vx <= vel_bound
            b_ineq[constraint_idx] = vel_bound
            constraint_idx += 1
            
            A_ineq[constraint_idx, state_idx + 2] = -1.0  # -vx <= vel_bound
            b_ineq[constraint_idx] = vel_bound
            constraint_idx += 1
            
            # 加速度限制: |ax|, |ay| <= acc_bound  
            A_ineq[constraint_idx, state_idx + 4] = 1.0   # ax <= acc_bound
            b_ineq[constraint_idx] = acc_bound
            constraint_idx += 1
            
            A_ineq[constraint_idx, state_idx + 4] = -1.0  # -ax <= acc_bound
            b_ineq[constraint_idx] = acc_bound
            constraint_idx += 1
        
        return A_ineq.tocsc(), b_ineq
    
    def _solve_qp(self, H, f, A_eq, b_eq, A_ineq, b_ineq) -> Dict:
        """求解QP问题"""
        if self.solver_type == QPSolver.OSQP and OSQP_AVAILABLE:
            return self._solve_with_osqp(H, f, A_eq, b_eq, A_ineq, b_ineq)
        elif self.solver_type == QPSolver.QPOASES and QPOASES_AVAILABLE:
            return self._solve_with_qpoases(H, f, A_eq, b_eq, A_ineq, b_ineq)
        else:
            return self._solve_with_scipy(H, f, A_eq, b_eq, A_ineq, b_ineq)
    
    def _solve_with_osqp(self, H, f, A_eq, b_eq, A_ineq, b_ineq) -> Dict:
        """使用OSQP求解器"""
        try:
            # 合并约束矩阵
            if A_eq.shape[0] > 0 and A_ineq.shape[0] > 0:
                A_combined = sp.vstack([A_eq, A_ineq])
                l_combined = np.concatenate([b_eq, -np.inf * np.ones(len(b_ineq))])
                u_combined = np.concatenate([b_eq, b_ineq])
            elif A_eq.shape[0] > 0:
                A_combined = A_eq
                l_combined = b_eq
                u_combined = b_eq
            else:
                A_combined = A_ineq
                l_combined = -np.inf * np.ones(len(b_ineq))
                u_combined = b_ineq
            
            # 创建OSQP问题
            prob = osqp.OSQP()
            prob.setup(P=H, q=f, A=A_combined, l=l_combined, u=u_combined, **self.solver_settings)
            
            # 求解
            result = prob.solve()
            
            return {
                'success': result.info.status == 'solved',
                'x': result.x if result.x is not None else np.zeros(H.shape[0]),
                'cost': result.info.obj_val if hasattr(result.info, 'obj_val') else float('inf'),
                'solver_info': {
                    'status': result.info.status,
                    'iterations': result.info.iter,
                    'solve_time': result.info.solve_time
                }
            }
            
        except Exception as e:
            print(f"OSQP求解失败: {e}")
            return {
                'success': False,
                'x': np.zeros(H.shape[0]),
                'cost': float('inf'),
                'solver_info': {'error': str(e)}
            }
    
    def _solve_with_scipy(self, H, f, A_eq, b_eq, A_ineq, b_ineq) -> Dict:
        """使用SciPy求解器（后备方案）"""
        from scipy.optimize import minimize
        
        try:
            n_vars = len(f)
            x0 = np.zeros(n_vars)
            
            # 目标函数
            def objective(x):
                return 0.5 * x.T @ H @ x + f.T @ x
            
            # 约束
            constraints = []
            
            # 等式约束
            if A_eq.shape[0] > 0:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: A_eq @ x - b_eq
                })
            
            # 不等式约束
            if A_ineq.shape[0] > 0:
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x: b_ineq - A_ineq @ x
                })
            
            # 求解
            result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                            options={'maxiter': 500, 'disp': False})
            
            return {
                'success': result.success,
                'x': result.x,
                'cost': result.fun,
                'solver_info': {
                    'status': 'solved' if result.success else 'failed',
                    'message': result.message
                }
            }
            
        except Exception as e:
            print(f"SciPy求解失败: {e}")
            return {
                'success': False,
                'x': np.zeros(len(f)),
                'cost': float('inf'),
                'solver_info': {'error': str(e)}
            }
    
    def _parse_solution(self, solution: Dict, current_state: LIPMState, gait_plan: GaitPlan) -> MPCResult:
        """解析优化结果"""
        result = MPCResult()
        result.success = solution['success']
        result.cost = solution['cost']
        result.solver_info = solution['solver_info']
        
        if not result.success:
            return result
        
        x = solution['x']
        dims = self._get_variable_dimensions(gait_plan)
        N = self.N
        
        # 1. 解析质心轨迹
        for k in range(N + 1):
            state_idx = k * 6
            
            com_state = LIPMState()
            com_state.com_position = Vector3D(
                x=x[state_idx], y=x[state_idx + 1], z=self.z_c
            )
            com_state.com_velocity = Vector3D(
                x=x[state_idx + 2], y=x[state_idx + 3], z=0.0
            )
            com_state.com_acceleration = Vector3D(
                x=x[state_idx + 4], y=x[state_idx + 5], z=0.0
            )
            com_state.timestamp = current_state.timestamp + k * self.dt
            
            result.predicted_com_states.append(com_state)
            result.com_position_trajectory.append(com_state.com_position)
            result.com_velocity_trajectory.append(com_state.com_velocity)
            result.com_acceleration_trajectory.append(com_state.com_acceleration)
        
        # 2. 解析接触力轨迹
        force_start_idx = dims['states']
        result.contact_forces = np.zeros((N, self.n_contacts, 3))
        
        for k in range(N):
            force_idx = force_start_idx + k * 6
            
            # 左脚力
            result.contact_forces[k, 0, :] = x[force_idx:force_idx+3]
            
            # 右脚力
            result.contact_forces[k, 1, :] = x[force_idx+3:force_idx+6]
        
        # 3. 当前时刻控制命令
        if N > 0:
            result.current_contact_forces = {
                'left': Vector3D(
                    x=result.contact_forces[0, 0, 0],
                    y=result.contact_forces[0, 0, 1], 
                    z=result.contact_forces[0, 0, 2]
                ),
                'right': Vector3D(
                    x=result.contact_forces[0, 1, 0],
                    y=result.contact_forces[0, 1, 1],
                    z=result.contact_forces[0, 1, 2]
                )
            }
            
            result.current_desired_com_acceleration = result.com_acceleration_trajectory[0]
        
        # 4. 解析ZMP轨迹
        zmp_start_idx = dims['states'] + dims['forces']
        for k in range(N):
            zmp_idx = zmp_start_idx + k * 2
            zmp_point = Vector3D(x=x[zmp_idx], y=x[zmp_idx + 1], z=0.0)
            result.zmp_trajectory.append(zmp_point)
        
        # 5. 下一步落脚位置（简化处理）
        if len(gait_plan.support_positions) > 0:
            next_support = gait_plan.support_positions[0]
            result.next_footstep = next_support.copy()
        
        return result
    
    def _get_fallback_result(self, current_state: LIPMState) -> MPCResult:
        """获取后备结果"""
        result = MPCResult()
        result.success = False
        result.cost = float('inf')
        result.solver_info = {'status': 'fallback'}
        
        # 提供零控制输入
        result.current_contact_forces = {
            'left': Vector3D(x=0.0, y=0.0, z=self.mass * self.g / 2),
            'right': Vector3D(x=0.0, y=0.0, z=self.mass * self.g / 2)
        }
        result.current_desired_com_acceleration = Vector3D(x=0.0, y=0.0, z=0.0)
        
        return result
    
    def get_solver_statistics(self) -> Dict:
        """获取求解器性能统计"""
        if not self.solve_times:
            return {}
        
        return {
            'average_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'success_rate': np.mean(self.success_rate),
            'total_solves': len(self.solve_times),
            'solver_type': self.solver_type.value
        }


def create_mpc_solver(lipm_model: SimplifiedDynamicsModel,
                     solver_type: QPSolver = QPSolver.OSQP,
                     prediction_horizon: int = 20,
                     dt: float = 0.1) -> MPCSolver:
    """
    创建MPC求解器实例
    
    Args:
        lipm_model: LIPM动力学模型
        solver_type: QP求解器类型
        prediction_horizon: 预测时域
        dt: 时间步长
        
    Returns:
        MPCSolver: MPC求解器实例
    """
    return MPCSolver(
        lipm_model=lipm_model,
        solver_type=solver_type,
        prediction_horizon=prediction_horizon,
        dt=dt
    ) 