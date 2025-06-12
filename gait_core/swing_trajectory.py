#!/usr/bin/env python3
"""
摆动足轨迹规划模块 - Swing Foot Trajectory Planner

实现多种摆动足轨迹规划方法，为四足机器人提供平滑的足部运动轨迹。

功能特性：
1. 三次多项式插值轨迹
2. 贝塞尔曲线轨迹
3. 正弦函数轨迹
4. 参数化轨迹配置
5. 水平方向平滑插值，竖直方向拱形轨迹
6. 可调节的轨迹参数

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import threading


class TrajectoryType(Enum):
    """轨迹类型枚举"""
    POLYNOMIAL = "polynomial"     # 三次多项式插值
    BEZIER = "bezier"            # 贝塞尔曲线
    SINUSOIDAL = "sinusoidal"    # 正弦函数
    CYCLOID = "cycloid"          # 摆线轨迹


@dataclass
class TrajectoryKeyPoint:
    """轨迹关键点"""
    time_ratio: float        # 时间比例 (0.0-1.0)
    position: np.ndarray     # 位置 [x, y, z]
    velocity: Optional[np.ndarray] = None  # 速度 [vx, vy, vz]
    acceleration: Optional[np.ndarray] = None  # 加速度 [ax, ay, az]


@dataclass
class SwingTrajectoryConfig:
    """摆动轨迹配置参数"""
    # 轨迹类型
    trajectory_type: TrajectoryType = TrajectoryType.BEZIER
    
    # 基本参数
    step_height: float = 0.08           # 抬脚高度 [m] (5-10cm范围)
    step_duration: float = 0.4          # 摆动周期 [s]
    ground_clearance: float = 0.02      # 地面间隙 [m]
    
    # 关键点时间比例
    lift_off_ratio: float = 0.0         # 离地时刻 (摆动开始)
    max_height_ratio: float = 0.5       # 最大高度时刻 (摆动中点)
    touch_down_ratio: float = 1.0       # 着地时刻 (摆动结束)
    
    # 贝塞尔曲线参数
    bezier_control_height_ratio: float = 0.8    # 控制点高度比例
    bezier_control_forward_ratio: float = 0.3   # 控制点前进比例
    
    # 多项式参数
    polynomial_order: int = 3           # 多项式阶数 (建议3次)
    
    # 正弦函数参数
    sin_vertical_periods: float = 1.0   # 垂直方向周期数
    sin_horizontal_smooth: bool = True  # 水平方向是否平滑
    
    # 速度约束
    max_vertical_velocity: float = 1.0  # 最大垂直速度 [m/s]
    max_horizontal_velocity: float = 2.0 # 最大水平速度 [m/s]
    
    # 安全参数
    velocity_smoothing: float = 0.1     # 速度平滑因子
    acceleration_limit: float = 10.0    # 加速度限制 [m/s²]


class SwingTrajectoryPlanner:
    """摆动足轨迹规划器"""
    
    def __init__(self, config: Optional[SwingTrajectoryConfig] = None):
        """初始化轨迹规划器"""
        self.config = config or SwingTrajectoryConfig()
        self._lock = threading.RLock()
        
        # 当前轨迹状态
        self.is_trajectory_active = False
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.end_position = np.array([0.0, 0.0, 0.0])
        self.trajectory_start_time = 0.0
        
        # 轨迹参数缓存
        self._cached_trajectory_params = None
        self._cache_valid = False
        
        print(f"SwingTrajectoryPlanner initialized with {self.config.trajectory_type.value} trajectory")
    
    def set_trajectory_parameters(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                step_height: Optional[float] = None,
                                step_duration: Optional[float] = None):
        """
        设置轨迹参数
        
        参数:
            start_pos: 起始位置 [x, y, z]
            end_pos: 结束位置 [x, y, z]
            step_height: 抬脚高度 [m]，可选
            step_duration: 摆动周期 [s]，可选
        """
        with self._lock:
            self.start_position = np.array(start_pos)
            self.end_position = np.array(end_pos)
            
            if step_height is not None:
                self.config.step_height = step_height
            if step_duration is not None:
                self.config.step_duration = step_duration
            
            # 清除缓存
            self._cache_valid = False
            self.is_trajectory_active = True
    
    def compute_trajectory_position(self, phase: float) -> np.ndarray:
        """
        计算给定相位的轨迹位置
        
        参数:
            phase: 轨迹相位 (0.0-1.0)
            
        返回:
            np.ndarray: 位置 [x, y, z]
        """
        # 限制相位范围
        phase = np.clip(phase, 0.0, 1.0)
        
        # 根据轨迹类型计算位置
        if self.config.trajectory_type == TrajectoryType.POLYNOMIAL:
            return self._compute_polynomial_trajectory(phase)
        elif self.config.trajectory_type == TrajectoryType.BEZIER:
            return self._compute_bezier_trajectory(phase)
        elif self.config.trajectory_type == TrajectoryType.SINUSOIDAL:
            return self._compute_sinusoidal_trajectory(phase)
        elif self.config.trajectory_type == TrajectoryType.CYCLOID:
            return self._compute_cycloid_trajectory(phase)
        else:
            # 默认使用贝塞尔曲线
            return self._compute_bezier_trajectory(phase)
    
    def compute_trajectory_velocity(self, phase: float, dt: float = 0.001) -> np.ndarray:
        """
        计算给定相位的轨迹速度
        
        参数:
            phase: 轨迹相位 (0.0-1.0)
            dt: 时间步长 [s]
            
        返回:
            np.ndarray: 速度 [vx, vy, vz]
        """
        if phase <= 0.0:
            # 起始速度
            pos_current = self.compute_trajectory_position(0.0)
            pos_next = self.compute_trajectory_position(dt / self.config.step_duration)
            velocity = (pos_next - pos_current) / dt
        elif phase >= 1.0:
            # 结束速度
            pos_prev = self.compute_trajectory_position(1.0 - dt / self.config.step_duration)
            pos_current = self.compute_trajectory_position(1.0)
            velocity = (pos_current - pos_prev) / dt
        else:
            # 中间速度（数值微分）
            pos_prev = self.compute_trajectory_position(phase - dt / self.config.step_duration)
            pos_next = self.compute_trajectory_position(phase + dt / self.config.step_duration)
            velocity = (pos_next - pos_prev) / (2 * dt)
        
        # 应用速度限制
        velocity = self._apply_velocity_limits(velocity)
        
        return velocity
    
    def _compute_polynomial_trajectory(self, phase: float) -> np.ndarray:
        """计算三次多项式轨迹"""
        # 水平方向 (X, Y)：三次多项式插值
        # 使用边界条件：起始和结束速度为0
        
        # 三次多项式: p(t) = at³ + bt² + ct + d
        # 边界条件: p(0)=p0, p(1)=p1, p'(0)=0, p'(1)=0
        # 解得: a=2(p0-p1), b=3(p1-p0), c=0, d=p0
        
        start_xy = self.start_position[:2]
        end_xy = self.end_position[:2]
        
        # 水平方向插值
        t = phase
        t2 = t * t
        t3 = t * t * t
        
        xy_position = start_xy + (3*t2 - 2*t3) * (end_xy - start_xy)
        
        # 竖直方向 (Z)：拱形轨迹
        z_position = self._compute_vertical_arch_trajectory(phase)
        
        return np.array([xy_position[0], xy_position[1], z_position])
    
    def _compute_bezier_trajectory(self, phase: float) -> np.ndarray:
        """计算贝塞尔曲线轨迹"""
        # 四点贝塞尔曲线：P0(起点) -> P1(控制点1) -> P2(控制点2) -> P3(终点)
        
        P0 = self.start_position.copy()
        P3 = self.end_position.copy()
        
        # 计算控制点
        direction = P3 - P0
        horizontal_distance = np.linalg.norm(direction[:2])
        
        # 控制点1：起点附近，稍微向上和向前
        P1 = P0.copy()
        P1[:2] += direction[:2] * self.config.bezier_control_forward_ratio
        P1[2] += self.config.step_height * self.config.bezier_control_height_ratio
        
        # 控制点2：终点附近，稍微向上和向后
        P2 = P3.copy()
        P2[:2] -= direction[:2] * self.config.bezier_control_forward_ratio
        P2[2] += self.config.step_height * self.config.bezier_control_height_ratio
        
        # 贝塞尔曲线公式: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        t = phase
        t2 = t * t
        t3 = t * t * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt * mt * mt
        
        position = (mt3 * P0 + 
                   3 * mt2 * t * P1 + 
                   3 * mt * t2 * P2 + 
                   t3 * P3)
        
        return position
    
    def _compute_sinusoidal_trajectory(self, phase: float) -> np.ndarray:
        """计算正弦函数轨迹"""
        # 水平方向：平滑插值或正弦插值
        if self.config.sin_horizontal_smooth:
            # 使用平滑插值（类似多项式）
            start_xy = self.start_position[:2]
            end_xy = self.end_position[:2]
            # 使用正弦函数的前半周期作为插值函数
            interpolation_factor = 0.5 * (1 - np.cos(np.pi * phase))
            xy_position = start_xy + interpolation_factor * (end_xy - start_xy)
        else:
            # 使用正弦波动
            start_xy = self.start_position[:2]
            end_xy = self.end_position[:2]
            base_xy = start_xy + phase * (end_xy - start_xy)
            # 添加小幅正弦波动
            wave_amplitude = 0.01  # 1cm波动
            perpendicular = np.array([-(end_xy[1] - start_xy[1]), end_xy[0] - start_xy[0]])
            if np.linalg.norm(perpendicular) > 1e-6:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                wave_offset = wave_amplitude * np.sin(2 * np.pi * phase) * perpendicular
                xy_position = base_xy + wave_offset
            else:
                xy_position = base_xy
        
        # 竖直方向：正弦拱形
        z_start = self.start_position[2]
        z_end = self.end_position[2]
        base_z = z_start + phase * (z_end - z_start)
        
        # 正弦抬脚轨迹
        sin_factor = np.sin(np.pi * phase * self.config.sin_vertical_periods)
        z_position = base_z + self.config.step_height * sin_factor
        
        return np.array([xy_position[0], xy_position[1], z_position])
    
    def _compute_cycloid_trajectory(self, phase: float) -> np.ndarray:
        """计算摆线轨迹"""
        # 摆线参数方程：x = r(θ - sin(θ)), y = r(1 - cos(θ))
        # 这里适配为足部轨迹
        
        start_xy = self.start_position[:2]
        end_xy = self.end_position[:2]
        
        # 水平方向：摆线的x分量（修改为适合足部运动）
        theta = np.pi * phase  # 半个摆线周期
        cycloid_x_factor = (theta - np.sin(theta)) / np.pi
        xy_position = start_xy + cycloid_x_factor * (end_xy - start_xy)
        
        # 竖直方向：摆线的y分量
        z_start = self.start_position[2]
        z_end = self.end_position[2]
        base_z = z_start + phase * (z_end - z_start)
        
        cycloid_z_factor = 1 - np.cos(theta)  # 0到2的范围
        z_position = base_z + self.config.step_height * cycloid_z_factor / 2
        
        return np.array([xy_position[0], xy_position[1], z_position])
    
    def _compute_vertical_arch_trajectory(self, phase: float) -> float:
        """计算竖直方向拱形轨迹"""
        z_start = self.start_position[2]
        z_end = self.end_position[2]
        
        # 基础高度插值
        base_z = z_start + phase * (z_end - z_start)
        
        # 拱形轨迹：使用简单的正弦函数形式，在摆动中点达到最大高度
        # 将相位映射到[0, π]区间，使得sin(π*phase)在phase=0.5时达到最大值1
        arch_height = self.config.step_height * np.sin(np.pi * phase)
        
        return base_z + arch_height
    
    def _apply_velocity_limits(self, velocity: np.ndarray) -> np.ndarray:
        """应用速度限制"""
        # 水平速度限制
        horizontal_vel = velocity[:2]
        horizontal_speed = np.linalg.norm(horizontal_vel)
        if horizontal_speed > self.config.max_horizontal_velocity:
            horizontal_vel = horizontal_vel * (self.config.max_horizontal_velocity / horizontal_speed)
            velocity[:2] = horizontal_vel
        
        # 垂直速度限制
        if abs(velocity[2]) > self.config.max_vertical_velocity:
            velocity[2] = np.sign(velocity[2]) * self.config.max_vertical_velocity
        
        return velocity
    
    def get_trajectory_info(self) -> Dict:
        """获取轨迹信息"""
        return {
            'trajectory_type': self.config.trajectory_type.value,
            'step_height': self.config.step_height,
            'step_duration': self.config.step_duration,
            'start_position': self.start_position.tolist(),
            'end_position': self.end_position.tolist(),
            'is_active': self.is_trajectory_active
        }
    
    def generate_trajectory_samples(self, num_samples: int = 100) -> Dict:
        """
        生成轨迹采样点（用于可视化和分析）
        
        参数:
            num_samples: 采样点数量
            
        返回:
            Dict: 包含位置、速度等信息的字典
        """
        phases = np.linspace(0.0, 1.0, num_samples)
        positions = []
        velocities = []
        
        for phase in phases:
            pos = self.compute_trajectory_position(phase)
            vel = self.compute_trajectory_velocity(phase)
            positions.append(pos)
            velocities.append(vel)
        
        return {
            'phases': phases.tolist(),
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'config': self.get_trajectory_info()
        }


class TrajectoryParameterOptimizer:
    """轨迹参数优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.optimization_history = []
    
    def optimize_step_height(self, terrain_difficulty: float, walking_speed: float) -> float:
        """
        根据地形和速度优化抬脚高度
        
        参数:
            terrain_difficulty: 地形难度 (0.0-1.0)
            walking_speed: 行走速度 [m/s]
            
        返回:
            float: 优化后的抬脚高度 [m]
        """
        # 基础抬脚高度
        base_height = 0.05  # 5cm
        
        # 根据地形调整
        terrain_adjustment = terrain_difficulty * 0.03  # 最多增加3cm
        
        # 根据速度调整
        speed_adjustment = min(walking_speed * 0.02, 0.02)  # 最多增加2cm
        
        optimized_height = base_height + terrain_adjustment + speed_adjustment
        
        # 限制范围在 3-12cm
        optimized_height = np.clip(optimized_height, 0.03, 0.12)
        
        return optimized_height
    
    def optimize_trajectory_timing(self, gait_frequency: float) -> Dict[str, float]:
        """
        根据步态频率优化轨迹时序
        
        参数:
            gait_frequency: 步态频率 [Hz]
            
        返回:
            Dict: 优化后的时序参数
        """
        # 摆动周期 = 步态周期 * 摆动占比
        swing_duty_factor = 0.4  # 摆动相占40%
        swing_duration = swing_duty_factor / gait_frequency
        
        # 优化关键点时间比例
        lift_off_ratio = 0.05      # 快速离地
        max_height_ratio = 0.45    # 中点稍早到达最高点
        touch_down_ratio = 0.95    # 留出缓冲时间
        
        return {
            'step_duration': swing_duration,
            'lift_off_ratio': lift_off_ratio,
            'max_height_ratio': max_height_ratio,
            'touch_down_ratio': touch_down_ratio
        }


# 工厂函数
def create_swing_trajectory_planner(trajectory_type: str = "bezier", 
                                  step_height: float = 0.08,
                                  step_duration: float = 0.4) -> SwingTrajectoryPlanner:
    """
    创建摆动轨迹规划器
    
    参数:
        trajectory_type: 轨迹类型 ("polynomial", "bezier", "sinusoidal", "cycloid")
        step_height: 抬脚高度 [m]
        step_duration: 摆动周期 [s]
        
    返回:
        SwingTrajectoryPlanner: 轨迹规划器实例
    """
    config = SwingTrajectoryConfig()
    
    # 设置轨迹类型
    if trajectory_type.lower() == "polynomial":
        config.trajectory_type = TrajectoryType.POLYNOMIAL
    elif trajectory_type.lower() == "bezier":
        config.trajectory_type = TrajectoryType.BEZIER
    elif trajectory_type.lower() == "sinusoidal":
        config.trajectory_type = TrajectoryType.SINUSOIDAL
    elif trajectory_type.lower() == "cycloid":
        config.trajectory_type = TrajectoryType.CYCLOID
    else:
        print(f"警告: 未知轨迹类型 '{trajectory_type}'，使用默认贝塞尔曲线")
        config.trajectory_type = TrajectoryType.BEZIER
    
    config.step_height = step_height
    config.step_duration = step_duration
    
    return SwingTrajectoryPlanner(config)


# 公共接口函数
def get_swing_foot_position(phase: float, start_pos: np.ndarray, target_pos: np.ndarray, 
                          step_height: float = 0.08, 
                          interpolation_type: str = "linear",
                          vertical_trajectory_type: str = "sine") -> np.ndarray:
    """
    计算摆动足在给定相位的期望位置
    
    参数:
        phase: 摆动相位 (0.0-1.0)，0.0为起点，1.0为终点
        start_pos: 起始位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
        step_height: 抬脚高度 [m]，默认0.08m (8cm)
        interpolation_type: 水平插值类型 ("linear", "cubic", "smooth")
        vertical_trajectory_type: 垂直轨迹类型 ("sine", "parabola", "smooth_parabola")
        
    返回:
        np.ndarray: 当前相位的足部期望位置 [x, y, z]
    """
    # 确保相位在有效范围内
    phase = np.clip(phase, 0.0, 1.0)
    
    # 转换为numpy数组
    start_pos = np.asarray(start_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    
    # 水平方向插值 (X, Y)
    horizontal_pos = _compute_horizontal_interpolation(
        phase, start_pos[:2], target_pos[:2], interpolation_type
    )
    
    # 垂直方向轨迹 (Z)
    vertical_pos = _compute_vertical_trajectory(
        phase, start_pos[2], target_pos[2], step_height, vertical_trajectory_type
    )
    
    # 组合结果
    return np.array([horizontal_pos[0], horizontal_pos[1], vertical_pos])


def _compute_horizontal_interpolation(phase: float, start_xy: np.ndarray, target_xy: np.ndarray, 
                                    interpolation_type: str) -> np.ndarray:
    """
    计算水平方向插值
    
    参数:
        phase: 相位 (0.0-1.0)
        start_xy: 起始平面坐标 [x, y]
        target_xy: 目标平面坐标 [x, y]
        interpolation_type: 插值类型
        
    返回:
        np.ndarray: 插值后的平面坐标 [x, y]
    """
    if interpolation_type == "linear":
        # 线性插值：pos_xy = start_xy + phase * (target_xy - start_xy)
        return start_xy + phase * (target_xy - start_xy)
    
    elif interpolation_type == "cubic":
        # 三次多项式插值，确保初末速度为零
        # 使用 Hermite 插值：p(t) = (2t³-3t²+1)p₀ + (t³-2t²+t)m₀ + (-2t³+3t²)p₁ + (t³-t²)m₁
        # 设置初末速度为零：m₀ = m₁ = 0
        # 简化为：p(t) = (2t³-3t²+1)p₀ + (-2t³+3t²)p₁ = p₀ + (3t²-2t³)(p₁-p₀)
        t = phase
        t2 = t * t
        t3 = t * t * t
        blend_factor = 3 * t2 - 2 * t3  # 平滑插值因子
        return start_xy + blend_factor * (target_xy - start_xy)
    
    elif interpolation_type == "smooth":
        # 基于正弦函数的平滑插值，确保平滑过渡
        # 使用 0.5 * (1 - cos(π * phase)) 作为插值因子
        blend_factor = 0.5 * (1 - np.cos(np.pi * phase))
        return start_xy + blend_factor * (target_xy - start_xy)
    
    else:
        # 默认使用线性插值
        return start_xy + phase * (target_xy - start_xy)


def _compute_vertical_trajectory(phase: float, start_z: float, target_z: float, 
                               step_height: float, trajectory_type: str) -> float:
    """
    计算垂直方向轨迹
    
    参数:
        phase: 相位 (0.0-1.0)
        start_z: 起始高度
        target_z: 目标高度
        step_height: 抬脚高度
        trajectory_type: 轨迹类型
        
    返回:
        float: 当前相位的垂直位置
    """
    # 基础高度插值
    base_z = start_z + phase * (target_z - start_z)
    
    if trajectory_type == "sine":
        # 正弦曲线：Z = h * sin(π * phase)
        # 在 phase=0 和 phase=1 时为0，phase=0.5 时达到最大值 h
        arch_height = step_height * np.sin(np.pi * phase)
        
    elif trajectory_type == "parabola":
        # 抛物线：Z = 4h * phase * (1 - phase)
        # 在 phase=0 和 phase=1 时为0，phase=0.5 时达到最大值 h
        arch_height = 4 * step_height * phase * (1 - phase)
        
    elif trajectory_type == "smooth_parabola":
        # 平滑抛物线，在两端有更平滑的过渡
        # 使用修正的抛物线公式
        if phase <= 0.1:
            # 起始段平滑过渡
            local_phase = phase / 0.1
            arch_height = step_height * 0.4 * local_phase * local_phase
        elif phase >= 0.9:
            # 结束段平滑过渡
            local_phase = (1.0 - phase) / 0.1
            arch_height = step_height * 0.4 * local_phase * local_phase
        else:
            # 中间段标准抛物线
            arch_height = 4 * step_height * phase * (1 - phase)
            
    else:
        # 默认使用正弦函数
        arch_height = step_height * np.sin(np.pi * phase)
    
    return base_z + arch_height


def get_swing_foot_velocity(phase: float, start_pos: np.ndarray, target_pos: np.ndarray,
                          step_height: float = 0.08, step_duration: float = 0.4,
                          interpolation_type: str = "linear", 
                          vertical_trajectory_type: str = "sine") -> np.ndarray:
    """
    计算摆动足在给定相位的期望速度（数值微分）
    
    参数:
        phase: 摆动相位 (0.0-1.0)
        start_pos: 起始位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
        step_height: 抬脚高度 [m]
        step_duration: 摆动周期 [s]
        interpolation_type: 水平插值类型
        vertical_trajectory_type: 垂直轨迹类型
        
    返回:
        np.ndarray: 当前相位的足部期望速度 [vx, vy, vz]
    """
    dt = 0.001  # 数值微分步长
    phase_dt = dt / step_duration  # 相位域的微分步长
    
    if phase <= 0.0:
        # 起始速度（前向差分）
        pos_current = get_swing_foot_position(0.0, start_pos, target_pos, 
                                            step_height, interpolation_type, vertical_trajectory_type)
        pos_next = get_swing_foot_position(phase_dt, start_pos, target_pos,
                                         step_height, interpolation_type, vertical_trajectory_type)
        velocity = (pos_next - pos_current) / dt
        
    elif phase >= 1.0:
        # 结束速度（后向差分）
        pos_prev = get_swing_foot_position(1.0 - phase_dt, start_pos, target_pos,
                                         step_height, interpolation_type, vertical_trajectory_type)
        pos_current = get_swing_foot_position(1.0, start_pos, target_pos,
                                            step_height, interpolation_type, vertical_trajectory_type)
        velocity = (pos_current - pos_prev) / dt
        
    else:
        # 中间速度（中心差分）
        pos_prev = get_swing_foot_position(phase - phase_dt, start_pos, target_pos,
                                         step_height, interpolation_type, vertical_trajectory_type)
        pos_next = get_swing_foot_position(phase + phase_dt, start_pos, target_pos,
                                         step_height, interpolation_type, vertical_trajectory_type)
        velocity = (pos_next - pos_prev) / (2 * dt)
    
    return velocity


def create_swing_foot_trajectory(start_pos: np.ndarray, target_pos: np.ndarray,
                                step_height: float = 0.08, num_points: int = 50,
                                interpolation_type: str = "cubic",
                                vertical_trajectory_type: str = "sine") -> Dict[str, np.ndarray]:
    """
    创建完整的摆动足轨迹
    
    参数:
        start_pos: 起始位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
        step_height: 抬脚高度 [m]
        num_points: 轨迹点数量
        interpolation_type: 水平插值类型
        vertical_trajectory_type: 垂直轨迹类型
        
    返回:
        Dict: 包含位置和相位信息的字典
    """
    phases = np.linspace(0.0, 1.0, num_points)
    positions = []
    
    for phase in phases:
        pos = get_swing_foot_position(phase, start_pos, target_pos, 
                                    step_height, interpolation_type, vertical_trajectory_type)
        positions.append(pos)
    
    return {
        'phases': phases,
        'positions': np.array(positions),
        'start_pos': start_pos,
        'target_pos': target_pos,
        'step_height': step_height,
        'interpolation_type': interpolation_type,
        'vertical_trajectory_type': vertical_trajectory_type
    }


if __name__ == "__main__":
    # 演示不同轨迹类型
    print("摆动足轨迹规划器演示")
    print("=" * 50)
    
    # 设置轨迹参数
    start_pos = np.array([0.0, 0.1, 0.0])  # 起始位置
    end_pos = np.array([0.15, 0.1, 0.0])   # 结束位置（步长15cm）
    step_height = 0.08                      # 8cm抬脚高度
    
    trajectory_types = ["polynomial", "bezier", "sinusoidal", "cycloid"]
    
    for traj_type in trajectory_types:
        print(f"\n{traj_type.upper()} 轨迹:")
        print("-" * 30)
        
        planner = create_swing_trajectory_planner(traj_type, step_height)
        planner.set_trajectory_parameters(start_pos, end_pos)
        
        # 采样关键点
        key_phases = [0.0, 0.25, 0.5, 0.75, 1.0]
        for phase in key_phases:
            pos = planner.compute_trajectory_position(phase)
            vel = planner.compute_trajectory_velocity(phase)
            print(f"  相位 {phase:.2f}: 位置({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                  f"速度({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")
    
    # 演示新的 get_swing_foot_position 函数
    print(f"\n使用 get_swing_foot_position 函数:")
    print("-" * 40)
    
    interpolation_types = ["linear", "cubic", "smooth"]
    vertical_types = ["sine", "parabola", "smooth_parabola"]
    
    for interp_type in interpolation_types:
        for vert_type in vertical_types:
            print(f"\n{interp_type.upper()} + {vert_type.upper()}:")
            
            # 测试关键点
            for phase in [0.0, 0.5, 1.0]:
                pos = get_swing_foot_position(phase, start_pos, end_pos, step_height, 
                                            interp_type, vert_type)
                print(f"  相位 {phase:.1f}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})") 