"""
DataBus - 机器人全局数据总线
定义机器人各模块共享的状态和指令数据结构

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import threading
import time


class ContactState(Enum):
    """足部接触状态枚举"""
    NO_CONTACT = 0      # 无接触
    CONTACT = 1         # 有接触
    SLIDING = 2         # 滑动接触
    UNKNOWN = -1        # 未知状态


class GaitPhase(Enum):
    """步态相位枚举"""
    STANCE = 0          # 支撑相
    SWING = 1           # 摆动相
    DOUBLE_SUPPORT = 2  # 双脚支撑
    FLIGHT = 3          # 飞行相（跳跃时）


class LegState(Enum): 
    """腿部状态枚举"""
    LEFT_LEG = 0
    RIGHT_LEG = 1


@dataclass
class Vector3D:
    """3D向量数据结构"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    def from_array(self, arr: np.ndarray):
        """从numpy数组设置值"""
        self.x, self.y, self.z = arr[0], arr[1], arr[2]


@dataclass
class Quaternion:
    """四元数数据结构"""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组 [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])
    
    def from_array(self, arr: np.ndarray):
        """从numpy数组设置值"""
        self.w, self.x, self.y, self.z = arr[0], arr[1], arr[2], arr[3]


@dataclass
class JointData:
    """单个关节数据"""
    name: str = ""                    # 关节名称
    position: float = 0.0             # 关节角度 [rad]
    velocity: float = 0.0             # 关节角速度 [rad/s]
    acceleration: float = 0.0         # 关节角加速度 [rad/s²]
    torque: float = 0.0              # 关节力矩 [Nm]
    effort: float = 0.0              # 实际输出力/力矩
    temperature: float = 0.0          # 电机温度 [°C]
    current: float = 0.0             # 电机电流 [A]
    voltage: float = 0.0             # 电机电压 [V]
    
    # 控制相关
    desired_position: float = 0.0     # 期望位置 [rad]
    desired_velocity: float = 0.0     # 期望速度 [rad/s]
    desired_torque: float = 0.0       # 期望力矩 [Nm]
    
    # 限制
    position_limit_min: float = -np.pi  # 位置下限
    position_limit_max: float = np.pi   # 位置上限
    velocity_limit: float = 10.0         # 速度限制
    torque_limit: float = 100.0          # 力矩限制


@dataclass
class EndEffectorState:
    """末端执行器（足部/手部）状态"""
    name: str = ""                    # 末端执行器名称
    position: Vector3D = field(default_factory=Vector3D)    # 位置 [m]
    velocity: Vector3D = field(default_factory=Vector3D)    # 速度 [m/s]
    acceleration: Vector3D = field(default_factory=Vector3D) # 加速度 [m/s²]
    orientation: Quaternion = field(default_factory=Quaternion)  # 姿态
    angular_velocity: Vector3D = field(default_factory=Vector3D) # 角速度 [rad/s]
    
    # 力信息
    force: Vector3D = field(default_factory=Vector3D)       # 接触力 [N]
    torque: Vector3D = field(default_factory=Vector3D)      # 接触力矩 [Nm]
    contact_state: ContactState = ContactState.UNKNOWN      # 接触状态
    contact_force_magnitude: float = 0.0                    # 接触力大小 [N]
    
    # 期望值
    desired_position: Vector3D = field(default_factory=Vector3D)
    desired_velocity: Vector3D = field(default_factory=Vector3D)
    desired_force: Vector3D = field(default_factory=Vector3D)


@dataclass
class IMUData:
    """IMU传感器数据"""
    # 姿态信息
    orientation: Quaternion = field(default_factory=Quaternion)  # 姿态四元数
    angular_velocity: Vector3D = field(default_factory=Vector3D) # 角速度 [rad/s]
    linear_acceleration: Vector3D = field(default_factory=Vector3D) # 线加速度 [m/s²]
    
    # 欧拉角表示
    roll: float = 0.0                 # 滚转角 [rad]
    pitch: float = 0.0                # 俯仰角 [rad]
    yaw: float = 0.0                  # 偏航角 [rad]


@dataclass
class CenterOfMassData:
    """质心数据"""
    position: Vector3D = field(default_factory=Vector3D)    # 质心位置 [m]
    velocity: Vector3D = field(default_factory=Vector3D)    # 质心速度 [m/s]
    acceleration: Vector3D = field(default_factory=Vector3D) # 质心加速度 [m/s²]


@dataclass
class GaitData:
    """步态数据（扩展版本）"""
    # 基础相位信息
    current_phase: GaitPhase = GaitPhase.STANCE              # 当前步态相位
    left_leg_phase: GaitPhase = GaitPhase.STANCE            # 左腿相位
    right_leg_phase: GaitPhase = GaitPhase.STANCE           # 右腿相位
    
    # 时间相关
    gait_cycle_time: float = 1.0                            # 步态周期时间 [s]
    phase_time: float = 0.0                                 # 当前相位时间 [s]
    cycle_phase: float = 0.0                                # 周期相位 (0.0-1.0)
    left_phase: float = 0.0                                 # 左腿相位 (0.0-1.0)
    right_phase: float = 0.0                                # 右腿相位 (0.0-1.0)
    
    # 基础步态参数（保持向后兼容）
    step_height: float = 0.05                               # 抬腿高度 [m]
    step_length: float = 0.1                                # 步长 [m]
    step_width: float = 0.2                                 # 步宽 [m]
    duty_factor: float = 0.6                                # 占空比（支撑相占比）
    walking_speed: float = 0.0                              # 行走速度 [m/s]
    turning_rate: float = 0.0                               # 转向速率 [rad/s]
    
    # 扩展步态参数
    swing_time: float = 0.4                                 # 摆动时间 [s]
    stance_time: float = 0.6                                # 支撑时间 [s]
    double_support_time: float = 0.2                        # 双支撑时间 [s]
    step_frequency: float = 1.0                             # 步频 [Hz]
    
    # 足部位置目标
    left_foot_target: Vector3D = field(default_factory=Vector3D)  # 左脚目标位置
    right_foot_target: Vector3D = field(default_factory=Vector3D) # 右脚目标位置
    
    # 质心轨迹参数
    com_height_target: float = 0.8                          # 目标质心高度 [m]
    com_lateral_shift_target: float = 0.0                   # 目标侧向偏移 [m]
    
    # 步态状态标志
    is_walking: bool = False                                # 是否在行走
    is_turning: bool = False                                # 是否在转向
    left_in_swing: bool = False                             # 左脚是否在摆动相
    right_in_swing: bool = False                            # 右脚是否在摆动相
    double_support: bool = True                             # 是否在双支撑相


class DataBus:
    """
    机器人全局数据总线
    
    功能：
    1. 存储机器人各模块共享的状态和指令
    2. 提供线程安全的数据访问接口
    3. 支持数据的读取和写入
    4. 提供数据验证和边界检查
    """
    
    def __init__(self):
        """初始化数据总线"""
        self._lock = threading.RLock()  # 递归锁，支持同一线程多次加锁
        self.timestamp = time.time()
        
        # ================== 关节数据 ==================
        # 根据AzureLoong机器人的关节配置定义
        self.joint_names = [
            # 手臂关节 (左)
            "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04",
            "J_arm_l_05", "J_arm_l_06", "J_arm_l_07",
            # 手臂关节 (右)
            "J_arm_r_01", "J_arm_r_02", "J_arm_r_03", "J_arm_r_04",
            "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
            # 头部关节
            "J_head_yaw", "J_head_pitch",
            # 腰部关节
            "J_waist_pitch", "J_waist_roll", "J_waist_yaw",
            # 左腿关节
            "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch",
            "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll",
            # 右腿关节
            "J_hip_r_roll", "J_hip_r_yaw", "J_hip_r_pitch",
            "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"
        ]
        
        # 初始化关节数据字典
        self.joints: Dict[str, JointData] = {}
        for name in self.joint_names:
            self.joints[name] = JointData(name=name)
        
        # ================== 传感器数据 ==================
        self.imu = IMUData()
        
        # ================== 末端执行器数据 ==================
        self.end_effectors: Dict[str, EndEffectorState] = {
            "left_foot": EndEffectorState(name="left_foot"),
            "right_foot": EndEffectorState(name="right_foot"),
            "left_hand": EndEffectorState(name="left_hand"),
            "right_hand": EndEffectorState(name="right_hand")
        }
        
        # ================== 质心数据 ==================
        self.center_of_mass = CenterOfMassData()
        
        # ================== 步态数据 ==================
        self.gait = GaitData()
        
        # ================== 步态参数管理器集成 ==================
        # 延迟导入避免循环依赖
        try:
            from gait_parameters import get_gait_manager
            self.gait_manager = get_gait_manager()
            self._has_gait_manager = True
        except ImportError:
            self.gait_manager = None
            self._has_gait_manager = False
            print("Warning: GaitParameterManager not available")
        
        # ================== 步态调度器集成 ==================
        try:
            from gait_scheduler import get_gait_scheduler, LegState
            self.gait_scheduler = get_gait_scheduler()
            self._has_gait_scheduler = True
            # 导入LegState用于数据总线
            self.LegState = LegState
        except ImportError:
            self.gait_scheduler = None
            self._has_gait_scheduler = False
            print("Warning: GaitScheduler not available")
        
        # ================== 步态状态字段 ==================
        self.legState = "DSt"                     # 当前支撑腿状态 (LSt/RSt/DSt/NSt)
        self.current_gait_state = "idle"          # 当前步态状态
        self.swing_leg = "none"                   # 当前摆动腿 (left/right/none/both)
        self.support_leg = "both"                 # 当前支撑腿 (left/right/both/none)
        self.state_transition_count = 0           # 状态转换次数
        self.last_state_transition_time = 0.0     # 上次状态转换时间
        
        # ================== 足步规划集成 ==================
        try:
            from foot_placement import get_foot_planner, Vector3D
            self.foot_planner = get_foot_planner()
            self._has_foot_planner = True
            # 导入Vector3D用于数据总线
            self.Vector3D = Vector3D
        except ImportError:
            self.foot_planner = None
            self._has_foot_planner = False
            print("Warning: FootPlacementPlanner not available")
        
        # ================== 足步目标位置字段 ==================
        self.target_foot_pos = {
            "left_foot": {"x": 0.0, "y": 0.09, "z": 0.0},   # 左脚目标位置
            "right_foot": {"x": 0.0, "y": -0.09, "z": 0.0}  # 右脚目标位置
        }
        self.current_foot_pos = {
            "left_foot": {"x": 0.0, "y": 0.09, "z": -0.6},   # 左脚当前位置
            "right_foot": {"x": 0.0, "y": -0.09, "z": -0.6}  # 右脚当前位置
        }
        self.foot_planning_count = 0              # 足步规划次数
        self.last_foot_planning_time = 0.0        # 上次足步规划时间
        
        # ================== 步态事件管理 ==================
        self.step_finished = False               # 步完成标志
        self.step_count = 0                      # 当前步数计数器
        self.total_steps = 0                     # 总步数
        self.left_step_count = 0                 # 左腿步数
        self.right_step_count = 0                # 右腿步数
        self.current_swing_leg = "none"          # 当前摆动腿
        self.last_swing_leg = "none"             # 上一个摆动腿
        self.swing_duration = 0.0                # 摆动持续时间
        self.step_completion_time = 0.0          # 步完成时间戳
        self.gait_phase_history = []             # 步态相位历史记录
        self.step_transition_events = []         # 步态转换事件列表
        self.step_completion_callbacks = []      # 步完成回调函数列表
        
        # ================== 控制输出 ==================
        self.control_torques: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.control_positions: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.control_velocities: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        
        # ================== 系统状态 ==================
        self.robot_mode = "IDLE"           # 机器人模式: IDLE, WALKING, RUNNING, STOP
        self.emergency_stop = False        # 急停状态
        self.battery_voltage = 0.0         # 电池电压 [V]
        self.battery_current = 0.0         # 电池电流 [A]
        self.system_temperature = 0.0      # 系统温度 [°C]
        
        # MPC专用数据结构
        self.desired_forces = {}              # MPC期望接触力 {foot_name: Vector3D}
        self.mpc_com_trajectory = []          # MPC质心轨迹缓存
        self.mpc_zmp_trajectory = []          # MPC ZMP轨迹缓存
        self.mpc_status = {                   # MPC状态信息
            'last_solve_time': 0.0,
            'solve_success': False,
            'cost': float('inf'),
            'solver_type': 'unknown'
        }
        
        print("DataBus initialized successfully")
    
    def update_timestamp(self):
        """更新时间戳"""
        with self._lock:
            self.timestamp = time.time()
    
    # ================== 关节数据接口 ==================
    def set_joint_position(self, joint_name: str, position: float) -> bool:
        """设置关节位置"""
        with self._lock:
            if joint_name in self.joints:
                self.joints[joint_name].position = position
                self.update_timestamp()
                return True
            return False
    
    def get_joint_position(self, joint_name: str) -> Optional[float]:
        """获取关节位置"""
        with self._lock:
            if joint_name in self.joints:
                return self.joints[joint_name].position
            return None
    
    def set_joint_velocity(self, joint_name: str, velocity: float) -> bool:
        """设置关节速度"""
        with self._lock:
            if joint_name in self.joints:
                self.joints[joint_name].velocity = velocity
                self.update_timestamp()
                return True
            return False
    
    def get_joint_velocity(self, joint_name: str) -> Optional[float]:
        """获取关节速度"""
        with self._lock:
            if joint_name in self.joints:
                return self.joints[joint_name].velocity
            return None
    
    def set_joint_torque(self, joint_name: str, torque: float) -> bool:
        """设置关节力矩"""
        with self._lock:
            if joint_name in self.joints:
                self.joints[joint_name].torque = torque
                self.update_timestamp()
                return True
            return False
    
    def get_joint_torque(self, joint_name: str) -> Optional[float]:
        """获取关节力矩"""
        with self._lock:
            if joint_name in self.joints:
                return self.joints[joint_name].torque
            return None
    
    def get_all_joint_positions(self) -> Dict[str, float]:
        """获取所有关节位置"""
        with self._lock:
            return {name: joint.position for name, joint in self.joints.items()}
    
    def get_all_joint_velocities(self) -> Dict[str, float]:
        """获取所有关节速度"""
        with self._lock:
            return {name: joint.velocity for name, joint in self.joints.items()}
    
    def get_all_joint_torques(self) -> Dict[str, float]:
        """获取所有关节力矩"""
        with self._lock:
            return {name: joint.torque for name, joint in self.joints.items()}
    
    # ================== 控制输出接口 ==================
    def set_control_torque(self, joint_name: str, torque: float) -> bool:
        """设置控制力矩输出"""
        with self._lock:
            if joint_name in self.control_torques:
                self.control_torques[joint_name] = torque
                self.update_timestamp()
                return True
            return False
    
    def get_control_torque(self, joint_name: str) -> Optional[float]:
        """获取控制力矩输出"""
        with self._lock:
            return self.control_torques.get(joint_name)
    
    def set_control_position(self, joint_name: str, position: float) -> bool:
        """设置控制位置输出"""
        with self._lock:
            if joint_name in self.control_positions:
                self.control_positions[joint_name] = position
                self.update_timestamp()
                return True
            return False
    
    def get_control_position(self, joint_name: str) -> Optional[float]:
        """获取控制位置输出"""
        with self._lock:
            return self.control_positions.get(joint_name)
    
    def get_all_control_torques(self) -> Dict[str, float]:
        """获取所有控制力矩输出"""
        with self._lock:
            return self.control_torques.copy()
    
    def get_all_control_positions(self) -> Dict[str, float]:
        """获取所有控制位置输出"""
        with self._lock:
            return self.control_positions.copy()
    
    # ================== 末端执行器接口 ==================
    def set_end_effector_position(self, ee_name: str, position: Vector3D) -> bool:
        """设置末端执行器位置"""
        with self._lock:
            if ee_name in self.end_effectors:
                self.end_effectors[ee_name].position = position
                self.update_timestamp()
                return True
            return False
    
    def get_end_effector_position(self, ee_name: str) -> Optional[Vector3D]:
        """获取末端执行器位置"""
        with self._lock:
            if ee_name in self.end_effectors:
                return self.end_effectors[ee_name].position
            return None
    
    def set_end_effector_contact_state(self, ee_name: str, contact_state: ContactState) -> bool:
        """设置末端执行器接触状态"""
        with self._lock:
            if ee_name in self.end_effectors:
                self.end_effectors[ee_name].contact_state = contact_state
                self.update_timestamp()
                return True
            return False
    
    def get_end_effector_contact_state(self, ee_name: str) -> Optional[ContactState]:
        """获取末端执行器接触状态"""
        with self._lock:
            if ee_name in self.end_effectors:
                return self.end_effectors[ee_name].contact_state
            return None
    
    def set_end_effector_contact_force(self, ee_name: str, force_magnitude: float) -> bool:
        """设置末端执行器接触力大小"""
        with self._lock:
            if ee_name in self.end_effectors:
                self.end_effectors[ee_name].contact_force_magnitude = force_magnitude
                # 根据力大小自动更新接触状态
                if force_magnitude > 20.0:  # 阈值可配置
                    self.end_effectors[ee_name].contact_state = ContactState.CONTACT
                else:
                    self.end_effectors[ee_name].contact_state = ContactState.NO_CONTACT
                self.update_timestamp()
                return True
            return False
    
    def get_end_effector_contact_force(self, ee_name: str) -> Optional[float]:
        """获取末端执行器接触力大小"""
        with self._lock:
            if ee_name in self.end_effectors:
                return self.end_effectors[ee_name].contact_force_magnitude
            return None
    
    # ================== IMU数据接口 ==================
    def set_imu_orientation(self, quaternion: Quaternion):
        """设置IMU姿态"""
        with self._lock:
            self.imu.orientation = quaternion
            self.update_timestamp()
    
    def get_imu_orientation(self) -> Quaternion:
        """获取IMU姿态"""
        with self._lock:
            return self.imu.orientation
    
    def set_imu_angular_velocity(self, angular_velocity: Vector3D):
        """设置IMU角速度"""
        with self._lock:
            self.imu.angular_velocity = angular_velocity
            self.update_timestamp()
    
    def get_imu_angular_velocity(self) -> Vector3D:
        """获取IMU角速度"""
        with self._lock:
            return self.imu.angular_velocity
    
    def set_imu_linear_acceleration(self, acceleration: Vector3D):
        """设置IMU线加速度"""
        with self._lock:
            self.imu.linear_acceleration = acceleration
            self.update_timestamp()
    
    def get_imu_linear_acceleration(self) -> Vector3D:
        """获取IMU线加速度"""
        with self._lock:
            return self.imu.linear_acceleration
    
    # ================== 质心数据接口 ==================
    def set_center_of_mass_position(self, position: Vector3D):
        """设置质心位置"""
        with self._lock:
            self.center_of_mass.position = position
            self.update_timestamp()
    
    def get_center_of_mass_position(self) -> Vector3D:
        """获取质心位置"""
        with self._lock:
            return self.center_of_mass.position
    
    def set_center_of_mass_velocity(self, velocity: Vector3D):
        """设置质心速度"""
        with self._lock:
            self.center_of_mass.velocity = velocity
            self.update_timestamp()
    
    def get_center_of_mass_velocity(self) -> Vector3D:
        """获取质心速度"""
        with self._lock:
            return self.center_of_mass.velocity
    
    # ================== 步态数据接口 ==================
    def set_gait_phase(self, phase: GaitPhase):
        """设置步态相位"""
        with self._lock:
            self.gait.current_phase = phase
            self.update_timestamp()
    
    def get_gait_phase(self) -> GaitPhase:
        """获取步态相位"""
        with self._lock:
            return self.gait.current_phase
    
    def set_leg_phase(self, leg: LegState, phase: GaitPhase):
        """设置腿部相位"""
        with self._lock:
            if leg == LegState.LEFT_LEG:
                self.gait.left_leg_phase = phase
            elif leg == LegState.RIGHT_LEG:
                self.gait.right_leg_phase = phase
            self.update_timestamp()
    
    def get_leg_phase(self, leg: LegState) -> GaitPhase:
        """获取腿部相位"""
        with self._lock:
            if leg == LegState.LEFT_LEG:
                return self.gait.left_leg_phase
            elif leg == LegState.RIGHT_LEG:
                return self.gait.right_leg_phase
            return GaitPhase.STANCE
    
    # ================== 系统状态接口 ==================
    def set_robot_mode(self, mode: str):
        """设置机器人模式"""
        with self._lock:
            self.robot_mode = mode
            self.update_timestamp()
    
    def get_robot_mode(self) -> str:
        """获取机器人模式"""
        with self._lock:
            return self.robot_mode
    
    def set_emergency_stop(self, stop: bool):
        """设置急停状态"""
        with self._lock:
            self.emergency_stop = stop
            self.update_timestamp()
    
    def get_emergency_stop(self) -> bool:
        """获取急停状态"""
        with self._lock:
            return self.emergency_stop
    
    # ================== 数据验证接口 ==================
    def validate_joint_limits(self, joint_name: str) -> bool:
        """验证关节限制"""
        with self._lock:
            if joint_name not in self.joints:
                return False
            
            joint = self.joints[joint_name]
            position_valid = (joint.position_limit_min <= joint.position <= joint.position_limit_max)
            velocity_valid = abs(joint.velocity) <= joint.velocity_limit
            torque_valid = abs(joint.torque) <= joint.torque_limit
            
            return position_valid and velocity_valid and torque_valid
    
    def get_system_status(self) -> Dict:
        """获取系统状态摘要"""
        with self._lock:
            return {
                "timestamp": self.timestamp,
                "robot_mode": self.robot_mode,
                "emergency_stop": self.emergency_stop,
                "battery_voltage": self.battery_voltage,
                "joint_count": len(self.joints),
                "end_effector_count": len(self.end_effectors),
                "gait_phase": self.gait.current_phase.name,
                "left_leg_phase": self.gait.left_leg_phase.name,
                "right_leg_phase": self.gait.right_leg_phase.name
            }
    
    def print_status(self):
        """打印系统状态"""
        status = self.get_system_status()
        print("=" * 50)
        print("DataBus System Status")
        print("=" * 50)
        for key, value in status.items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    # ================== 步态参数接口 ==================
    def update_gait_from_manager(self):
        """从步态参数管理器更新步态数据"""
        if not self._has_gait_manager:
            return
        
        with self._lock:
            manager = self.gait_manager
            
            # 更新时间参数
            self.gait.gait_cycle_time = manager.timing.gait_cycle_time
            self.gait.swing_time = manager.timing.swing_time
            self.gait.stance_time = manager.timing.stance_time
            self.gait.double_support_time = manager.timing.double_support_time
            self.gait.step_frequency = manager.timing.step_frequency
            
            # 更新空间参数
            self.gait.step_length = manager.spatial.step_length
            self.gait.step_width = manager.spatial.step_width
            self.gait.step_height = manager.spatial.step_height
            
            # 更新动力学参数
            self.gait.walking_speed = manager.dynamics.walking_speed
            self.gait.com_height_target = manager.dynamics.com_height
            
            # 更新占空比
            self.gait.duty_factor = manager.timing.stance_ratio
            
            self.update_timestamp()
    
    def update_gait_phase(self, current_time: float):
        """更新步态相位信息"""
        if not self._has_gait_manager:
            return
        
        with self._lock:
            # 获取相位信息
            phase_info = self.gait_manager.get_phase_info(current_time)
            
            # 更新相位数据
            self.gait.cycle_phase = phase_info["cycle_phase"]
            self.gait.left_phase = phase_info["left_phase"]
            self.gait.right_phase = phase_info["right_phase"]
            self.gait.left_in_swing = phase_info["left_in_swing"]
            self.gait.right_in_swing = phase_info["right_in_swing"]
            self.gait.double_support = phase_info["double_support"]
            
            # 更新腿部相位枚举
            if self.gait.left_in_swing:
                self.gait.left_leg_phase = GaitPhase.SWING
            else:
                self.gait.left_leg_phase = GaitPhase.STANCE
                
            if self.gait.right_in_swing:
                self.gait.right_leg_phase = GaitPhase.SWING
            else:
                self.gait.right_leg_phase = GaitPhase.STANCE
            
            # 计算足部目标位置
            left_pos = self.gait_manager.calculate_foot_placement("left", self.gait.left_phase)
            right_pos = self.gait_manager.calculate_foot_placement("right", self.gait.right_phase)
            
            self.gait.left_foot_target.x, self.gait.left_foot_target.y, self.gait.left_foot_target.z = left_pos
            self.gait.right_foot_target.x, self.gait.right_foot_target.y, self.gait.right_foot_target.z = right_pos
            
            self.update_timestamp()
    
    def set_walking_speed(self, speed: float):
        """设置行走速度"""
        if self._has_gait_manager:
            self.gait_manager.set_walking_speed(speed)
            self.update_gait_from_manager()
        else:
            with self._lock:
                self.gait.walking_speed = speed
                self.update_timestamp()
    
    def get_walking_speed(self) -> float:
        """获取行走速度"""
        with self._lock:
            return self.gait.walking_speed
    
    def set_step_frequency(self, frequency: float):
        """设置步频"""
        if self._has_gait_manager:
            self.gait_manager.set_step_frequency(frequency)
            self.update_gait_from_manager()
        else:
            with self._lock:
                self.gait.step_frequency = frequency
                self.gait.gait_cycle_time = 1.0 / frequency if frequency > 0 else 1.0
                self.update_timestamp()
    
    def get_step_frequency(self) -> float:
        """获取步频"""
        with self._lock:
            return self.gait.step_frequency
    
    def load_gait_preset(self, preset_name: str) -> bool:
        """加载步态参数预设"""
        if self._has_gait_manager:
            success = self.gait_manager.load_preset(preset_name)
            if success:
                self.update_gait_from_manager()
            return success
        else:
            print("Warning: GaitParameterManager not available")
            return False
    
    def get_foot_target_position(self, foot_name: str) -> Optional[Vector3D]:
        """获取足部目标位置"""
        with self._lock:
            if foot_name == "left_foot" or foot_name == "left":
                return self.gait.left_foot_target
            elif foot_name == "right_foot" or foot_name == "right":
                return self.gait.right_foot_target
            return None
    
    def is_foot_in_swing(self, foot_name: str) -> bool:
        """检查足部是否在摆动相"""
        with self._lock:
            if foot_name == "left_foot" or foot_name == "left":
                return self.gait.left_in_swing
            elif foot_name == "right_foot" or foot_name == "right":
                return self.gait.right_in_swing
            return False
    
    def get_gait_phase_info(self) -> Dict:
        """获取完整的步态相位信息"""
        with self._lock:
            return {
                "cycle_phase": self.gait.cycle_phase,
                "left_phase": self.gait.left_phase,
                "right_phase": self.gait.right_phase,
                "left_in_swing": self.gait.left_in_swing,
                "right_in_swing": self.gait.right_in_swing,
                "double_support": self.gait.double_support,
                "gait_cycle_time": self.gait.gait_cycle_time,
                "step_frequency": self.gait.step_frequency,
                "walking_speed": self.gait.walking_speed
            }
    
    def start_walking(self):
        """开始行走"""
        with self._lock:
            self.gait.is_walking = True
            self.set_robot_mode("WALKING")
            self.update_timestamp()
    
    def stop_walking(self):
        """停止行走"""
        with self._lock:
            self.gait.is_walking = False
            self.gait.walking_speed = 0.0
            self.set_robot_mode("IDLE")
            if self._has_gait_manager:
                self.gait_manager.dynamics.walking_speed = 0.0
            self.update_timestamp()
    
    def start_turning(self, turning_rate: float):
        """开始转向"""
        with self._lock:
            self.gait.is_turning = True
            self.gait.turning_rate = turning_rate
            if self._has_gait_manager:
                self.gait_manager.dynamics.turning_speed = turning_rate
            self.update_timestamp()
    
    def stop_turning(self):
        """停止转向"""
        with self._lock:
            self.gait.is_turning = False
            self.gait.turning_rate = 0.0
            if self._has_gait_manager:
                self.gait_manager.dynamics.turning_speed = 0.0
            self.update_timestamp()
    
    def get_available_gait_presets(self) -> List[str]:
        """获取可用的步态预设列表"""
        if self._has_gait_manager:
            return list(self.gait_manager.parameter_presets.keys())
        else:
            return []
    
    def export_gait_parameters(self, filename: str):
        """导出步态参数到文件"""
        if self._has_gait_manager:
            self.gait_manager.export_to_file(filename)
        else:
            print("Warning: GaitParameterManager not available")
    
    def import_gait_parameters(self, filename: str):
        """从文件导入步态参数"""
        if self._has_gait_manager:
            self.gait_manager.import_from_file(filename)
            self.update_gait_from_manager()
        else:
            print("Warning: GaitParameterManager not available")
    
    # ================== 步态调度器接口 ==================
    def update_gait_scheduler(self, dt: float) -> bool:
        """
        更新步态调度器状态
        
        参数:
            dt: 时间步长 [s]
            
        返回:
            bool: 是否发生状态转换
        """
        if not self._has_gait_scheduler:
            return False
        
        with self._lock:
            # 获取足部力和速度数据
            left_force = self.end_effectors["left_foot"].contact_force_magnitude
            right_force = self.end_effectors["right_foot"].contact_force_magnitude
            left_velocity = self.end_effectors["left_foot"].velocity.to_array()
            right_velocity = self.end_effectors["right_foot"].velocity.to_array()
            
            # 更新调度器传感器数据
            self.gait_scheduler.update_sensor_data(
                left_force, right_force, left_velocity, right_velocity
            )
            
            # 更新状态机
            state_changed = self.gait_scheduler.update_gait_state(dt)
            
            if state_changed:
                # 同步状态到数据总线
                self._sync_scheduler_state()
                self.state_transition_count += 1
                self.last_state_transition_time = time.time()
            
            self.update_timestamp()
            return state_changed
    
    def _sync_scheduler_state(self):
        """同步调度器状态到数据总线"""
        if not self._has_gait_scheduler:
            return
        
        scheduler = self.gait_scheduler
        
        # 更新基本状态字段
        self.legState = scheduler.leg_state.value
        self.current_gait_state = scheduler.current_state.value
        self.swing_leg = scheduler.swing_leg
        self.support_leg = scheduler.support_leg
        
        # 更新步态数据结构中的状态信息
        if scheduler.swing_leg == "left":
            self.gait.left_in_swing = True
            self.gait.right_in_swing = False
        elif scheduler.swing_leg == "right":
            self.gait.left_in_swing = False
            self.gait.right_in_swing = True
        elif scheduler.swing_leg == "both":
            self.gait.left_in_swing = True
            self.gait.right_in_swing = True
        else:  # none
            self.gait.left_in_swing = False
            self.gait.right_in_swing = False
        
        # 更新双支撑状态
        self.gait.double_support = (scheduler.leg_state == self.LegState.DSt)
    
    def start_gait_scheduler_walking(self):
        """通过调度器开始行走"""
        if self._has_gait_scheduler:
            self.gait_scheduler.start_walking()
            self._sync_scheduler_state()
            self.update_timestamp()
        else:
            # 回退到原有方法
            self.start_walking()
    
    def stop_gait_scheduler_walking(self):
        """通过调度器停止行走"""
        if self._has_gait_scheduler:
            self.gait_scheduler.stop_walking()
            self._sync_scheduler_state()
            self.update_timestamp()
        else:
            # 回退到原有方法
            self.stop_walking()
    
    def emergency_stop_gait(self):
        """步态紧急停止"""
        if self._has_gait_scheduler:
            self.gait_scheduler.emergency_stop()
            self._sync_scheduler_state()
            self.set_emergency_stop(True)
            self.update_timestamp()
    
    def reset_gait_scheduler(self):
        """重置步态调度器"""
        if self._has_gait_scheduler:
            self.gait_scheduler.reset()
            self._sync_scheduler_state()
            self.state_transition_count = 0
            self.last_state_transition_time = 0.0
            self.update_timestamp()
    
    def get_gait_state_info(self) -> Dict:
        """获取步态状态信息"""
        with self._lock:
            info = {
                "leg_state": self.legState,
                "current_gait_state": self.current_gait_state,
                "swing_leg": self.swing_leg,
                "support_leg": self.support_leg,
                "state_transition_count": self.state_transition_count,
                "last_transition_time": self.last_state_transition_time,
                "has_scheduler": self._has_gait_scheduler
            }
            
            if self._has_gait_scheduler:
                state_info = self.gait_scheduler.get_current_state_info()
                info.update({
                    "current_state_duration": state_info.duration,
                    "total_scheduler_time": self.gait_scheduler.total_time,
                    "left_foot_force": self.gait_scheduler.left_foot_force,
                    "right_foot_force": self.gait_scheduler.right_foot_force,
                    "left_foot_contact": self.gait_scheduler.left_foot_contact,
                    "right_foot_contact": self.gait_scheduler.right_foot_contact
                })
            
            return info
    
    def get_gait_state_statistics(self) -> Dict:
        """获取步态状态统计"""
        if self._has_gait_scheduler:
            return self.gait_scheduler.get_state_statistics()
        else:
            return {"error": "GaitScheduler not available"}
    
    def set_gait_scheduler_config(self, config_dict: Dict):
        """设置步态调度器配置"""
        if not self._has_gait_scheduler:
            return False
        
        # 更新配置参数
        config = self.gait_scheduler.config
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return True
    
    def add_gait_state_callback(self, callback):
        """添加步态状态变化回调"""
        if self._has_gait_scheduler:
            self.gait_scheduler.add_state_change_callback(callback)
    
    def remove_gait_state_callback(self, callback):
        """移除步态状态变化回调"""
        if self._has_gait_scheduler:
            self.gait_scheduler.remove_state_change_callback(callback)
    
    def is_in_swing_phase(self, leg: str) -> bool:
        """检查指定腿是否在摆动相"""
        with self._lock:
            if leg.lower() in ["left", "left_foot"]:
                return self.swing_leg in ["left", "both"]
            elif leg.lower() in ["right", "right_foot"]:
                return self.swing_leg in ["right", "both"]
            return False
    
    def is_in_support_phase(self, leg: str) -> bool:
        """检查指定腿是否在支撑相"""
        with self._lock:
            if leg.lower() in ["left", "left_foot"]:
                return self.support_leg in ["left", "both"]
            elif leg.lower() in ["right", "right_foot"]:
                return self.support_leg in ["right", "both"]
            return False
    
    def is_in_double_support(self) -> bool:
        """检查是否在双支撑相"""
        with self._lock:
            return self.legState == "DSt"
    
    def get_current_support_leg(self) -> str:
        """获取当前主要支撑腿"""
        with self._lock:
            if self.legState == "LSt":
                return "left"
            elif self.legState == "RSt":
                return "right"
            elif self.legState == "DSt":
                return "both"
            else:  # NSt
                return "none"
    
    def get_current_swing_leg(self) -> str:
        """获取当前摆动腿"""
        with self._lock:
            return self.swing_leg
    
    def print_gait_status(self):
        """打印步态状态"""
        state_info = self.get_gait_state_info()
        print(f"\n=== 步态状态总览 ===")
        print(f"支撑腿状态: {state_info['leg_state']}")
        print(f"当前步态状态: {state_info['current_gait_state']}")
        print(f"摆动腿: {state_info['swing_leg']}")
        print(f"支撑腿: {state_info['support_leg']}")
        print(f"状态转换次数: {state_info['state_transition_count']}")
        
        if state_info['has_scheduler']:
            print(f"当前状态持续时间: {state_info['current_state_duration']:.3f}s")
            print(f"调度器总运行时间: {state_info['total_scheduler_time']:.3f}s")
            print(f"左脚接触力: {state_info['left_foot_force']:.1f}N")
            print(f"右脚接触力: {state_info['right_foot_force']:.1f}N")
            print(f"左脚接触: {state_info['left_foot_contact']}")
            print(f"右脚接触: {state_info['right_foot_contact']}")
        
        print("=" * 30)
    
    # ================== 足步规划接口 ==================
    def update_foot_kinematics(self):
        """更新足部运动学计算"""
        if not self._has_foot_planner:
            return False
        
        with self._lock:
            # 获取当前关节角度
            joint_angles = self.get_all_joint_positions()
            
            # 更新足步规划器中的足部状态
            self.foot_planner.update_foot_states_from_kinematics(joint_angles)
            
            # 同步足部位置到数据总线
            current_positions = self.foot_planner.get_current_foot_positions()
            
            for foot_name, position in current_positions.items():
                if foot_name in self.current_foot_pos:
                    self.current_foot_pos[foot_name] = {
                        "x": position.x,
                        "y": position.y,
                        "z": position.z
                    }
            
            self.update_timestamp()
            return True
    
    def trigger_foot_placement_planning(self, swing_leg: str, support_leg: str):
        """
        触发足步规划计算
        
        参数:
            swing_leg: 摆动腿标识 ("left"/"right")
            support_leg: 支撑腿标识 ("left"/"right")
        """
        if not self._has_foot_planner:
            return None
        
        with self._lock:
            # 首先更新足部运动学
            self.update_foot_kinematics()
            
            # 设置运动意图 (从步态参数获取)
            if self._has_gait_manager:
                velocity = self.Vector3D(
                    self.gait_manager.get_step_frequency() * self.gait_manager.get_step_length(),
                    0.0,  # 暂时不考虑横向速度
                    0.0
                )
                self.foot_planner.set_body_motion_intent(velocity, 0.0)
            
            # 执行足步规划
            target_position = self.foot_planner.plan_foot_placement(swing_leg, support_leg)
            
            # 更新数据总线中的目标位置
            foot_key = f"{swing_leg.lower()}_foot"
            if foot_key in self.target_foot_pos:
                self.target_foot_pos[foot_key] = {
                    "x": target_position.x,
                    "y": target_position.y,
                    "z": target_position.z
                }
            
            # 更新统计信息
            self.foot_planning_count += 1
            self.last_foot_planning_time = time.time()
            
            self.update_timestamp()
            return target_position
    
    def set_body_motion_intent(self, forward_velocity: float, lateral_velocity: float = 0.0, 
                              heading: float = 0.0):
        """
        设置身体运动意图
        
        参数:
            forward_velocity: 前进速度 [m/s]
            lateral_velocity: 横向速度 [m/s] 
            heading: 身体朝向角度 [rad]
        """
        if self._has_foot_planner:
            velocity = self.Vector3D(forward_velocity, lateral_velocity, 0.0)
            self.foot_planner.set_body_motion_intent(velocity, heading)
    
    def update_gait_scheduler_with_foot_planning(self, dt: float) -> bool:
        """
        更新步态调度器并自动触发足步规划
        
        参数:
            dt: 时间步长 [s]
            
        返回:
            bool: 是否发生状态转换
        """
        if not self._has_gait_scheduler:
            return False
        
        # 首先更新足部运动学
        self.update_foot_kinematics()
        
        # 记录转换前的状态
        old_state = self.current_gait_state
        old_swing_leg = self.swing_leg
        
        # 更新步态调度器
        state_changed = self.update_gait_scheduler(dt)
        
        # 如果状态发生转换且有新的摆动腿，触发足步规划
        if state_changed and self.swing_leg != "none" and self.swing_leg != old_swing_leg:
            if self.swing_leg in ["left", "right"]:
                # 确定支撑腿
                support_leg = "right" if self.swing_leg == "left" else "left"
                
                # 触发足步规划
                target_pos = self.trigger_foot_placement_planning(self.swing_leg, support_leg)
                
                if target_pos and self.config.get('enable_logging', True):
                    print(f"[足步规划] {self.swing_leg}腿摆动，目标位置: "
                          f"({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")
        
        return state_changed
    
    def get_target_foot_position(self, foot: str) -> Dict[str, float]:
        """
        获取指定足的目标位置
        
        参数:
            foot: 足部标识 ("left"/"left_foot"/"right"/"right_foot")
            
        返回:
            Dict: 目标位置 {"x": float, "y": float, "z": float}
        """
        with self._lock:
            foot_key = foot.lower()
            if not foot_key.endswith("_foot"):
                foot_key += "_foot"
            
            return self.target_foot_pos.get(foot_key, {"x": 0.0, "y": 0.0, "z": 0.0}).copy()
    
    def get_current_foot_position(self, foot: str) -> Dict[str, float]:
        """
        获取指定足的当前位置
        
        参数:
            foot: 足部标识 ("left"/"left_foot"/"right"/"right_foot")
            
        返回:
            Dict: 当前位置 {"x": float, "y": float, "z": float}
        """
        with self._lock:
            foot_key = foot.lower()
            if not foot_key.endswith("_foot"):
                foot_key += "_foot"
            
            return self.current_foot_pos.get(foot_key, {"x": 0.0, "y": 0.0, "z": 0.0}).copy()
    
    def get_foot_displacement(self, foot: str) -> Dict[str, float]:
        """
        获取指定足的位移 (目标位置 - 当前位置)
        
        参数:
            foot: 足部标识
            
        返回:
            Dict: 位移向量 {"x": float, "y": float, "z": float}
        """
        current = self.get_current_foot_position(foot)
        target = self.get_target_foot_position(foot)
        
        return {
            "x": target["x"] - current["x"],
            "y": target["y"] - current["y"],
            "z": target["z"] - current["z"]
        }
    
    def get_foot_distance_to_target(self, foot: str) -> float:
        """
        获取指定足到目标位置的距离
        
        参数:
            foot: 足部标识
            
        返回:
            float: 距离 [m]
        """
        displacement = self.get_foot_displacement(foot)
        return np.sqrt(displacement["x"]**2 + displacement["y"]**2 + displacement["z"]**2)
    
    def set_foot_planning_config(self, config_dict: Dict):
        """
        设置足步规划配置
        
        参数:
            config_dict: 配置参数字典
        """
        if not self._has_foot_planner:
            return False
        
        # 更新配置参数
        config = self.foot_planner.config
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return True
    
    def set_terrain_type(self, terrain_type: str):
        """
        设置地形类型
        
        参数:
            terrain_type: 地形类型 ("flat"/"slope"/"stairs"/"rough"/"unknown")
        """
        if self._has_foot_planner:
            from foot_placement import TerrainType
            
            terrain_map = {
                "flat": TerrainType.FLAT,
                "slope": TerrainType.SLOPE,
                "stairs": TerrainType.STAIRS,
                "rough": TerrainType.ROUGH,
                "unknown": TerrainType.UNKNOWN
            }
            
            if terrain_type.lower() in terrain_map:
                self.foot_planner.set_terrain_type(terrain_map[terrain_type.lower()])
    
    def set_planning_strategy(self, strategy: str):
        """
        设置足步规划策略
        
        参数:
            strategy: 规划策略 ("static_walk"/"dynamic_walk"/"adaptive"/"terrain_adaptive"/"stabilizing")
        """
        if self._has_foot_planner:
            from foot_placement import FootPlacementStrategy
            
            strategy_map = {
                "static_walk": FootPlacementStrategy.STATIC_WALK,
                "dynamic_walk": FootPlacementStrategy.DYNAMIC_WALK,
                "adaptive": FootPlacementStrategy.ADAPTIVE,
                "terrain_adaptive": FootPlacementStrategy.TERRAIN_ADAPTIVE,
                "stabilizing": FootPlacementStrategy.STABILIZING
            }
            
            if strategy.lower() in strategy_map:
                self.foot_planner.set_planning_strategy(strategy_map[strategy.lower()])
    
    def get_foot_planning_info(self) -> Dict:
        """获取足步规划信息"""
        with self._lock:
            info = {
                "has_foot_planner": self._has_foot_planner,
                "foot_planning_count": self.foot_planning_count,
                "last_foot_planning_time": self.last_foot_planning_time,
                "target_positions": self.target_foot_pos.copy(),
                "current_positions": self.current_foot_pos.copy()
            }
            
            if self._has_foot_planner:
                planner_stats = self.foot_planner.get_planning_statistics()
                info.update({
                    "planning_strategy": planner_stats["current_strategy"],
                    "terrain_type": planner_stats["terrain_type"],
                    "planner_count": planner_stats["planning_count"]
                })
            
            return info
    
    def print_foot_planning_status(self):
        """打印足步规划状态"""
        info = self.get_foot_planning_info()
        
        print(f"\n=== 足步规划状态 ===")
        print(f"足步规划器可用: {info['has_foot_planner']}")
        
        if info['has_foot_planner']:
            print(f"规划策略: {info.get('planning_strategy', 'N/A')}")
            print(f"地形类型: {info.get('terrain_type', 'N/A')}")
            print(f"数据总线规划次数: {info['foot_planning_count']}")
            print(f"规划器总次数: {info.get('planner_count', 0)}")
            print(f"上次规划时间: {info['last_foot_planning_time']:.3f}")
            
            print(f"\n当前足部位置:")
            for foot, pos in info['current_positions'].items():
                print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            
            print(f"\n目标足部位置:")
            for foot, pos in info['target_positions'].items():
                print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
            
            print(f"\n足部位移:")
            for foot in ["left_foot", "right_foot"]:
                displacement = self.get_foot_displacement(foot)
                distance = self.get_foot_distance_to_target(foot)
                print(f"  {foot}: 位移({displacement['x']:.3f}, {displacement['y']:.3f}, {displacement['z']:.3f}) "
                      f"距离: {distance:.3f}m")
        
        print("=" * 30)
    
    # ================== 步态事件管理方法 ==================
    
    def mark_step_completion(self, completed_swing_leg: str, swing_duration: float, next_swing_leg: str):
        """
        标记步完成事件
        
        参数:
            completed_swing_leg: 完成摆动的腿 ("left"/"right")
            swing_duration: 摆动持续时间 [s]
            next_swing_leg: 下一个摆动腿 ("left"/"right"/"none")
        """
        with self._lock:
            current_time = time.time()
            
            # 更新步态事件标志
            self.step_finished = True
            self.step_completion_time = current_time
            self.last_swing_leg = self.current_swing_leg
            self.current_swing_leg = completed_swing_leg
            self.swing_duration = swing_duration
            
            # 更新步数计数器
            self.step_count += 1
            self.total_steps += 1
            
            if completed_swing_leg == "left":
                self.left_step_count += 1
            elif completed_swing_leg == "right":
                self.right_step_count += 1
            
            # 记录步完成事件
            event = {
                "timestamp": current_time,
                "event_type": "step_completion",
                "completed_swing_leg": completed_swing_leg,
                "swing_duration": swing_duration,
                "next_swing_leg": next_swing_leg,
                "step_number": self.step_count,
                "gait_state": self.current_gait_state
            }
            
            self.step_transition_events.append(event)
            
            # 限制事件历史长度
            if len(self.step_transition_events) > 100:
                self.step_transition_events.pop(0)
            
            # 记录相位历史
            phase_record = {
                "timestamp": current_time,
                "swing_leg": completed_swing_leg,
                "duration": swing_duration,
                "step_count": self.step_count
            }
            
            self.gait_phase_history.append(phase_record)
            
            # 限制历史长度
            if len(self.gait_phase_history) > 50:
                self.gait_phase_history.pop(0)
            
            # 调用回调函数
            for callback in self.step_completion_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Warning: 步完成回调执行失败: {e}")
    
    def reset_step_completion_flag(self):
        """重置步完成标志"""
        with self._lock:
            self.step_finished = False
    
    def is_step_completed(self) -> bool:
        """检查是否有步完成"""
        with self._lock:
            return self.step_finished
    
    def get_step_count(self) -> int:
        """获取当前步数"""
        with self._lock:
            return self.step_count
    
    def get_total_steps(self) -> int:
        """获取总步数"""
        with self._lock:
            return self.total_steps
    
    def get_step_statistics(self) -> Dict:
        """获取步数统计信息"""
        with self._lock:
            return {
                "step_count": self.step_count,
                "total_steps": self.total_steps,
                "left_step_count": self.left_step_count,
                "right_step_count": self.right_step_count,
                "current_swing_leg": self.current_swing_leg,
                "last_swing_leg": self.last_swing_leg,
                "swing_duration": self.swing_duration,
                "step_completion_time": self.step_completion_time,
                "step_finished": self.step_finished
            }
    
    def reset_step_counters(self):
        """重置步数计数器"""
        with self._lock:
            self.step_count = 0
            self.left_step_count = 0
            self.right_step_count = 0
            self.step_finished = False
            self.step_transition_events.clear()
            self.gait_phase_history.clear()
    
    def add_step_completion_callback(self, callback):
        """添加步完成回调函数"""
        with self._lock:
            if callback not in self.step_completion_callbacks:
                self.step_completion_callbacks.append(callback)
    
    def remove_step_completion_callback(self, callback):
        """移除步完成回调函数"""
        with self._lock:
            if callback in self.step_completion_callbacks:
                self.step_completion_callbacks.remove(callback)
    
    def get_recent_step_events(self, count: int = 10) -> List[Dict]:
        """获取最近的步态事件"""
        with self._lock:
            return self.step_transition_events[-count:] if self.step_transition_events else []
    
    def get_gait_phase_history(self, count: int = 10) -> List[Dict]:
        """获取步态相位历史"""
        with self._lock:
            return self.gait_phase_history[-count:] if self.gait_phase_history else []
    
    def print_step_status(self):
        """打印步态状态信息"""
        with self._lock:
            print("\n" + "="*50 + " 步态事件状态 " + "="*50)
            
            # 步数统计
            stats = self.get_step_statistics()
            print(f"当前步数: {stats['step_count']}")
            print(f"总步数: {stats['total_steps']}")
            print(f"左腿步数: {stats['left_step_count']}")
            print(f"右腿步数: {stats['right_step_count']}")
            print(f"当前摆动腿: {stats['current_swing_leg']}")
            print(f"上一摆动腿: {stats['last_swing_leg']}")
            print(f"摆动持续时间: {stats['swing_duration']:.3f}s")
            print(f"步完成标志: {stats['step_finished']}")
            
            # 最近事件
            recent_events = self.get_recent_step_events(5)
            if recent_events:
                print(f"\n最近5个步态事件:")
                for event in recent_events:
                    print(f"  步数{event['step_number']}: {event['completed_swing_leg']}腿摆动完成 "
                          f"({event['swing_duration']:.3f}s) -> {event['next_swing_leg']}腿")
            
            # 相位历史
            phase_history = self.get_gait_phase_history(5)
            if phase_history:
                print(f"\n最近5个相位记录:")
                for record in phase_history:
                    print(f"  步数{record['step_count']}: {record['swing_leg']}腿 "
                          f"({record['duration']:.3f}s)")
            
            print("="*120)
    
    # ================== 模块间数据接口 ==================
    
    def get_gait_data_for_trajectory_planning(self) -> Dict:
        """
        为足部轨迹规划模块提供步态数据
        
        返回:
            Dict: 轨迹规划所需的步态信息
        """
        with self._lock:
            if self._has_gait_scheduler:
                scheduler_data = self.gait_scheduler.get_gait_state_data()
                timing_info = self.gait_scheduler.get_timing_info()
                leg_states = self.gait_scheduler.get_leg_states()
                
                return {
                    # 当前状态
                    "current_state": scheduler_data["current_state"],
                    "swing_leg": scheduler_data["swing_leg"],
                    "support_leg": scheduler_data["support_leg"],
                    
                    # 时间信息
                    "swing_progress": timing_info["swing_progress"],
                    "cycle_phase": timing_info["cycle_phase"],
                    "time_to_swing_end": timing_info["time_to_swing_end"],
                    "is_swing_phase": timing_info["is_swing_phase"],
                    
                    # 目标位置
                    "target_foot_positions": self.target_foot_pos.copy(),
                    "current_foot_positions": self.current_foot_pos.copy(),
                    
                    # 步态参数
                    "swing_time": scheduler_data["swing_time"],
                    "step_height": self.gait.step_height,
                    "step_length": self.gait.step_length
                }
            return {}
    
    def get_gait_data_for_mpc(self) -> Dict:
        """
        为MPC控制器提供步态数据
        
        返回:
            Dict: MPC控制所需的步态信息
        """
        with self._lock:
            if self._has_gait_scheduler:
                scheduler_data = self.gait_scheduler.get_gait_state_data()
                timing_info = self.gait_scheduler.get_timing_info()
                prediction = self.gait_scheduler.get_next_swing_prediction()
                
                return {
                    # 当前支撑状态
                    "support_leg": scheduler_data["support_leg"],
                    "in_double_support": scheduler_data["support_leg"] == "both",
                    "left_is_support": scheduler_data["support_leg"] in ["left", "both"],
                    "right_is_support": scheduler_data["support_leg"] in ["right", "both"],
                    
                    # 时间信息
                    "current_time": timing_info["current_time"],
                    "swing_time_remaining": timing_info["time_to_swing_end"],
                    "cycle_phase": timing_info["cycle_phase"],
                    
                    # 预测信息
                    "next_swing_leg": prediction["next_swing_leg"],
                    "time_to_next_swing": prediction["time_to_next_swing"],
                    "predicted_swing_start": prediction["predicted_swing_start_time"],
                    
                    # 质心目标
                    "com_height_target": self.gait.com_height_target,
                    "com_lateral_shift": self.gait.com_lateral_shift_target,
                    
                    # 接触力信息
                    "left_foot_force": self.end_effectors["left_foot"].contact_force_magnitude,
                    "right_foot_force": self.end_effectors["right_foot"].contact_force_magnitude
                }
            return {}
    
    def get_sensor_data_for_gait_scheduler(self) -> Dict:
        """
        为步态调度器提供传感器数据
        
        返回:
            Dict: 步态调度器所需的传感器信息
        """
        with self._lock:
            return {
                # 足部力传感器
                "left_foot_force": self.end_effectors["left_foot"].contact_force_magnitude,
                "right_foot_force": self.end_effectors["right_foot"].contact_force_magnitude,
                "left_foot_contact": self.end_effectors["left_foot"].contact_state.value == 1,
                "right_foot_contact": self.end_effectors["right_foot"].contact_state.value == 1,
                
                # 足部速度
                "left_foot_velocity": self.end_effectors["left_foot"].velocity.to_array(),
                "right_foot_velocity": self.end_effectors["right_foot"].velocity.to_array(),
                
                # IMU数据
                "body_acceleration": self.imu.linear_acceleration.to_array(),
                "body_angular_velocity": self.imu.angular_velocity.to_array(),
                "body_orientation": self.imu.orientation.to_array(),
                
                # 质心状态
                "com_position": self.center_of_mass.position.to_array(),
                "com_velocity": self.center_of_mass.velocity.to_array()
            }
    
    def update_gait_targets_from_scheduler(self):
        """
        从步态调度器更新目标位置和状态
        """
        if self._has_gait_scheduler:
            try:
                # 更新目标足部位置
                target_positions = self.gait_scheduler.get_target_foot_positions()
                self.target_foot_pos.update(target_positions)
                
                # 更新步态状态
                leg_states = self.gait_scheduler.get_leg_states()
                self.swing_leg = leg_states["swing_leg"]
                self.support_leg = leg_states["support_leg"]
                self.legState = leg_states["leg_state"]
                
                # 更新步态参数
                timing_info = self.gait_scheduler.get_timing_info()
                self.gait.cycle_phase = timing_info["cycle_phase"]
                self.gait.phase_time = timing_info["swing_time"]
                self.gait.left_in_swing = leg_states["left_is_swing"]
                self.gait.right_in_swing = leg_states["right_is_swing"]
                self.gait.double_support = leg_states["in_double_support"]
                
            except Exception as e:
                print(f"Warning: 从步态调度器更新目标失败: {e}")
    
    def set_external_sensor_data(self, sensor_data: Dict):
        """
        设置来自外部系统的传感器数据
        
        参数:
            sensor_data: 传感器数据字典
        """
        with self._lock:
            # 更新足部传感器数据
            if "left_foot_force" in sensor_data:
                self.end_effectors["left_foot"].contact_force_magnitude = sensor_data["left_foot_force"]
            if "right_foot_force" in sensor_data:
                self.end_effectors["right_foot"].contact_force_magnitude = sensor_data["right_foot_force"]
            
            if "left_foot_velocity" in sensor_data:
                self.end_effectors["left_foot"].velocity.from_array(sensor_data["left_foot_velocity"])
            if "right_foot_velocity" in sensor_data:
                self.end_effectors["right_foot"].velocity.from_array(sensor_data["right_foot_velocity"])
            
            # 更新IMU数据
            if "body_acceleration" in sensor_data:
                self.imu.linear_acceleration.from_array(sensor_data["body_acceleration"])
            if "body_angular_velocity" in sensor_data:
                self.imu.angular_velocity.from_array(sensor_data["body_angular_velocity"])
    
    def get_interface_summary(self) -> Dict:
        """
        获取数据接口摘要信息
        
        返回:
            Dict: 接口状态摘要
        """
        with self._lock:
            return {
                "data_bus_timestamp": self.timestamp,
                "has_gait_scheduler": self._has_gait_scheduler,
                "has_gait_manager": self._has_gait_manager,
                "has_foot_planner": self._has_foot_planner,
                
                # 模块状态
                "gait_scheduler_state": self.current_gait_state if self._has_gait_scheduler else "unavailable",
                "step_count": self.step_count,
                "foot_planning_count": self.foot_planning_count,
                
                # 数据完整性
                "joint_count": len(self.joints),
                "end_effector_count": len(self.end_effectors),
                "has_imu_data": bool(self.imu),
                "has_com_data": bool(self.center_of_mass),
                
                # 接口可用性
                "trajectory_interface_ready": self._has_gait_scheduler and self._has_foot_planner,
                "mpc_interface_ready": self._has_gait_scheduler,
                "sensor_interface_ready": True
            }
    
    def set_desired_contact_force(self, foot_name: str, force: Vector3D) -> bool:
        """
        设置足部期望接触力（MPC输出）
        
        Args:
            foot_name: 足部名称 ('left_foot', 'right_foot')
            force: 期望接触力向量
            
        Returns:
            bool: 设置是否成功
        """
        with self._lock:
            try:
                self.desired_forces[foot_name] = force
                self.update_timestamp()
                return True
            except Exception as e:
                print(f"设置期望接触力失败: {e}")
                return False
    
    def get_desired_contact_force(self, foot_name: str) -> Optional[Vector3D]:
        """
        获取足部期望接触力
        
        Args:
            foot_name: 足部名称
            
        Returns:
            Vector3D: 期望接触力，如果不存在返回None
        """
        with self._lock:
            return self.desired_forces.get(foot_name, None)
    
    def get_all_desired_contact_forces(self) -> Dict[str, Vector3D]:
        """获取所有足部的期望接触力"""
        with self._lock:
            return self.desired_forces.copy()
    
    def set_center_of_mass_acceleration(self, acceleration: Vector3D):
        """
        设置质心加速度（MPC输出）
        
        Args:
            acceleration: 质心加速度向量
        """
        with self._lock:
            self.center_of_mass.acceleration = acceleration
            self.update_timestamp()
    
    def get_center_of_mass_acceleration(self) -> Vector3D:
        """获取质心加速度"""
        with self._lock:
            return self.center_of_mass.acceleration
    
    def set_mpc_com_trajectory(self, trajectory: List[Vector3D]):
        """
        设置MPC质心轨迹预测
        
        Args:
            trajectory: 质心位置轨迹列表
        """
        with self._lock:
            self.mpc_com_trajectory = trajectory.copy()
            self.update_timestamp()
    
    def get_mpc_com_trajectory(self) -> List[Vector3D]:
        """获取MPC质心轨迹预测"""
        with self._lock:
            return self.mpc_com_trajectory.copy()
    
    def set_mpc_zmp_trajectory(self, zmp_trajectory: List[Vector3D]):
        """
        设置MPC ZMP轨迹预测
        
        Args:
            zmp_trajectory: ZMP轨迹列表
        """
        with self._lock:
            self.mpc_zmp_trajectory = zmp_trajectory.copy()
            self.update_timestamp()
    
    def get_mpc_zmp_trajectory(self) -> List[Vector3D]:
        """获取MPC ZMP轨迹预测"""
        with self._lock:
            return self.mpc_zmp_trajectory.copy()
    
    def update_mpc_status(self, solve_time: float, success: bool, cost: float, solver_type: str):
        """
        更新MPC求解状态
        
        Args:
            solve_time: 求解时间
            success: 求解是否成功
            cost: 优化代价
            solver_type: 求解器类型
        """
        with self._lock:
            self.mpc_status.update({
                'last_solve_time': solve_time,
                'solve_success': success,
                'cost': cost,
                'solver_type': solver_type,
                'timestamp': time.time()
            })
            self.update_timestamp()
    
    def get_mpc_status(self) -> Dict:
        """获取MPC状态信息"""
        with self._lock:
            return self.mpc_status.copy()
    
    def set_target_foot_position_vector3d(self, foot_name: str, position: Vector3D) -> bool:
        """
        设置足部目标位置（Vector3D版本）
        
        Args:
            foot_name: 足部名称
            position: 目标位置向量
            
        Returns:
            bool: 设置是否成功
        """
        with self._lock:
            try:
                # 标准化足部名称，与get_target_foot_position保持一致
                foot_key = foot_name.lower()
                if not foot_key.endswith("_foot"):
                    foot_key += "_foot"
                
                self.target_foot_pos[foot_key] = {
                    'x': position.x,
                    'y': position.y,
                    'z': position.z
                }
                self.update_timestamp()
                return True
            except Exception as e:
                print(f"设置足部目标位置失败: {e}")
                return False
    
    def clear_mpc_data(self):
        """清除MPC相关数据"""
        with self._lock:
            self.desired_forces.clear()
            self.mpc_com_trajectory.clear()
            self.mpc_zmp_trajectory.clear()
            # 注意：不清除target_foot_pos，因为这是共享的步态数据
            self.mpc_status = {
                'last_solve_time': 0.0,
                'solve_success': False,
                'cost': float('inf'),
                'solver_type': 'unknown'
            }
            self.update_timestamp()
    
    def get_mpc_data_summary(self) -> Dict:
        """获取MPC数据摘要"""
        with self._lock:
            return {
                'desired_forces_count': len(self.desired_forces),
                'com_trajectory_length': len(self.mpc_com_trajectory),
                'zmp_trajectory_length': len(self.mpc_zmp_trajectory),
                'mpc_status': self.mpc_status.copy(),
                'target_foot_positions': self.target_foot_pos.copy()
            }


# 全局数据总线实例
# 这是整个机器人系统共享的单一数据总线实例
global_data_bus = DataBus()


def get_data_bus() -> DataBus:
    """获取全局数据总线实例"""
    return global_data_bus


if __name__ == "__main__":
    # 测试代码
    bus = get_data_bus()
    
    # 测试关节数据设置和获取
    bus.set_joint_position("J_hip_l_roll", 0.5)
    position = bus.get_joint_position("J_hip_l_roll")
    print(f"Left hip roll position: {position}")
    
    # 测试末端执行器数据
    left_foot_pos = Vector3D(0.0, 0.1, -1.0)
    bus.set_end_effector_position("left_foot", left_foot_pos)
    
    # 测试步态数据
    bus.set_gait_phase(GaitPhase.SWING)
    bus.set_leg_phase(LegState.LEFT_LEG, GaitPhase.SWING)
    
    # 打印系统状态
    bus.print_status() 