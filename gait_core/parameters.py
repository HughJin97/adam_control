"""
步态参数管理模块 - Gait Parameters Management

定义和管理机器人行走步态的所有关键参数，包括：
1. 步态周期时间参数
2. 空间位置参数  
3. 动力学参数
4. 稳定性参数

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import json
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time


class GaitType(Enum):
    """步态类型枚举"""
    STATIC_WALK = "static_walk"      # 静态步行（始终至少一只脚着地）
    DYNAMIC_WALK = "dynamic_walk"    # 动态步行（允许短暂双脚离地）
    TROT = "trot"                    # 小跑步态
    RUN = "run"                      # 跑步步态
    STAND = "stand"                  # 静止站立
    BALANCE = "balance"              # 平衡调整


class FootPhase(Enum):
    """单脚相位枚举"""
    STANCE = 0                       # 支撑相
    LIFT_OFF = 1                     # 离地瞬间
    SWING = 2                        # 摆动相
    TOUCH_DOWN = 3                   # 着地瞬间


@dataclass
class TimingParameters:
    """步态时间参数"""
    # 基础周期参数
    gait_cycle_time: float = 1.0     # 完整步态周期时间 [s]
    swing_time: float = 0.4          # 单脚摆动时间 [s] 
    stance_time: float = 0.6         # 单脚支撑时间 [s]
    double_support_time: float = 0.2 # 双脚支撑时间 [s]
    
    # 相位比例（占整个周期的比例）
    swing_ratio: float = 0.4         # 摆动相比例
    stance_ratio: float = 0.6        # 支撑相比例
    double_support_ratio: float = 0.2 # 双支撑相比例
    
    # 步频相关
    step_frequency: float = 1.0      # 步频 [Hz] (每秒步数)
    cadence: float = 120.0           # 步伐节拍 [steps/min]
    
    def update_from_cycle_time(self):
        """根据周期时间更新其他时间参数"""
        self.swing_time = self.gait_cycle_time * self.swing_ratio
        self.stance_time = self.gait_cycle_time * self.stance_ratio
        self.double_support_time = self.gait_cycle_time * self.double_support_ratio
        self.step_frequency = 1.0 / self.gait_cycle_time
        self.cadence = 60.0 / self.gait_cycle_time
    
    def update_from_frequency(self, frequency: float):
        """根据步频更新时间参数"""
        self.step_frequency = frequency
        self.gait_cycle_time = 1.0 / frequency
        self.cadence = frequency * 60.0
        self.update_from_cycle_time()


@dataclass
class SpatialParameters:
    """步态空间参数"""
    # 步长参数
    step_length: float = 0.15        # 前进步长 [m]
    step_width: float = 0.12         # 步宽（足部横向间距）[m] 
    step_height: float = 0.05        # 抬脚高度 [m]
    
    # 髋关节几何参数
    hip_width: float = 0.18          # 髋关节间距 [m]
    hip_height: float = 0.8          # 髋关节高度 [m]
    leg_length: float = 0.8          # 腿长 [m]
    
    # 足迹参数
    foot_length: float = 0.25        # 足部长度 [m]
    foot_width: float = 0.1          # 足部宽度 [m]
    
    # 运动包络
    max_step_length: float = 0.4     # 最大步长 [m]
    max_step_width: float = 0.3      # 最大步宽 [m]
    max_step_height: float = 0.15    # 最大抬脚高度 [m]
    
    # 转向参数
    turning_radius: float = 1.0      # 转向半径 [m]
    max_turning_rate: float = 0.5    # 最大转向速率 [rad/s]
    
    def validate_parameters(self) -> bool:
        """验证参数有效性"""
        if self.step_length > self.max_step_length:
            return False
        if self.step_width > self.max_step_width:
            return False
        if self.step_height > self.max_step_height:
            return False
        return True


@dataclass
class DynamicsParameters:
    """步态动力学参数"""
    # 速度参数
    walking_speed: float = 0.15      # 行走速度 [m/s]
    max_walking_speed: float = 1.0   # 最大行走速度 [m/s]
    acceleration: float = 0.5        # 加速度 [m/s²]
    deceleration: float = 0.8        # 减速度 [m/s²]
    
    # 角速度参数
    turning_speed: float = 0.0       # 转向速度 [rad/s]
    max_turning_speed: float = 1.0   # 最大转向速度 [rad/s]
    angular_acceleration: float = 2.0 # 角加速度 [rad/s²]
    
    # 质心轨迹参数
    com_height: float = 0.8          # 期望质心高度 [m]
    com_oscillation_amplitude: float = 0.02  # 质心振荡幅度 [m]
    com_lateral_shift: float = 0.02  # 质心侧向偏移 [m]
    
    # 足部轨迹参数
    foot_clearance: float = 0.05     # 足部离地间隙 [m]
    foot_landing_velocity: float = 0.1 # 足部着地速度 [m/s]
    
    def calculate_step_length_from_speed(self, cycle_time: float) -> float:
        """根据行走速度计算步长"""
        return self.walking_speed * cycle_time
    
    def calculate_speed_from_step_params(self, step_length: float, cycle_time: float) -> float:
        """根据步长和周期时间计算速度"""
        return step_length / cycle_time


@dataclass 
class StabilityParameters:
    """步态稳定性参数"""
    # ZMP相关参数
    zmp_margin_x: float = 0.05       # ZMP前后安全边界 [m]
    zmp_margin_y: float = 0.03       # ZMP左右安全边界 [m]
    
    # 平衡控制参数
    balance_gain_x: float = 1.0      # 前后平衡增益
    balance_gain_y: float = 1.0      # 左右平衡增益
    balance_gain_yaw: float = 0.5    # 偏航平衡增益
    
    # 姿态控制参数
    max_body_roll: float = 0.1       # 最大身体滚转角 [rad]
    max_body_pitch: float = 0.1      # 最大身体俯仰角 [rad]
    max_body_yaw: float = 0.2        # 最大身体偏航角 [rad]
    
    # 支撑多边形参数
    support_polygon_margin: float = 0.02  # 支撑多边形安全边界 [m]
    
    # 动态稳定性参数
    capture_point_gain: float = 1.0  # 捕获点控制增益
    step_adjustment_gain: float = 0.5 # 步长调整增益


@dataclass
class FootTrajectoryParameters:
    """足部轨迹参数"""
    # 摆动相轨迹参数
    swing_trajectory_type: str = "bezier"  # 轨迹类型: "bezier", "polynomial", "cycloidal"
    
    # 贝塞尔曲线参数（如果使用贝塞尔轨迹）
    bezier_control_height_ratio: float = 0.7  # 控制点高度比例
    bezier_control_forward_ratio: float = 0.3 # 控制点前进比例
    
    # 多项式轨迹参数
    polynomial_order: int = 5        # 多项式阶数
    
    # 摆动相关键点时间比例
    lift_off_time_ratio: float = 0.1    # 离地时间比例
    mid_swing_time_ratio: float = 0.5   # 摆动中点时间比例  
    touch_down_time_ratio: float = 0.9  # 着地时间比例
    
    # 着地缓冲参数
    landing_damping: float = 0.8     # 着地阻尼系数
    landing_stiffness: float = 1000.0 # 着地刚度系数


class GaitParameterManager:
    """步态参数管理器"""
    
    def __init__(self):
        """初始化步态参数管理器"""
        self._lock = threading.RLock()
        
        # 初始化参数组
        self.timing = TimingParameters()
        self.spatial = SpatialParameters()
        self.dynamics = DynamicsParameters()
        self.stability = StabilityParameters()
        self.trajectory = FootTrajectoryParameters()
        
        # 当前步态类型
        self.current_gait_type = GaitType.STATIC_WALK
        
        # 参数预设
        self.parameter_presets: Dict[str, Dict] = {}
        self._create_default_presets()
        
        print("GaitParameterManager initialized")
    
    def _create_default_presets(self):
        """创建默认参数预设"""
        # 慢速步行预设
        self.parameter_presets["slow_walk"] = {
            "timing": {
                "gait_cycle_time": 2.0,
                "swing_ratio": 0.4,
                "stance_ratio": 0.6,
                "double_support_ratio": 0.3
            },
            "spatial": {
                "step_length": 0.1,
                "step_width": 0.12,
                "step_height": 0.03
            },
            "dynamics": {
                "walking_speed": 0.05,
                "com_height": 0.8
            }
        }
        
        # 正常步行预设
        self.parameter_presets["normal_walk"] = {
            "timing": {
                "gait_cycle_time": 1.2,
                "swing_ratio": 0.4,
                "stance_ratio": 0.6,
                "double_support_ratio": 0.2
            },
            "spatial": {
                "step_length": 0.15,
                "step_width": 0.12,
                "step_height": 0.05
            },
            "dynamics": {
                "walking_speed": 0.125,
                "com_height": 0.8
            }
        }
        
        # 快速步行预设
        self.parameter_presets["fast_walk"] = {
            "timing": {
                "gait_cycle_time": 0.8,
                "swing_ratio": 0.45,
                "stance_ratio": 0.55,
                "double_support_ratio": 0.1
            },
            "spatial": {
                "step_length": 0.2,
                "step_width": 0.1,
                "step_height": 0.08
            },
            "dynamics": {
                "walking_speed": 0.25,
                "com_height": 0.85
            }
        }
        
        # 原地踏步预设
        self.parameter_presets["march_in_place"] = {
            "timing": {
                "gait_cycle_time": 1.0,
                "swing_ratio": 0.5,
                "stance_ratio": 0.5,
                "double_support_ratio": 0.0
            },
            "spatial": {
                "step_length": 0.0,
                "step_width": 0.12,
                "step_height": 0.1
            },
            "dynamics": {
                "walking_speed": 0.0,
                "com_height": 0.8
            }
        }
    
    def load_preset(self, preset_name: str) -> bool:
        """加载参数预设"""
        with self._lock:
            if preset_name not in self.parameter_presets:
                print(f"Warning: Preset '{preset_name}' not found")
                return False
            
            preset = self.parameter_presets[preset_name]
            
            # 更新各参数组
            if "timing" in preset:
                for key, value in preset["timing"].items():
                    if hasattr(self.timing, key):
                        setattr(self.timing, key, value)
                self.timing.update_from_cycle_time()
            
            if "spatial" in preset:
                for key, value in preset["spatial"].items():
                    if hasattr(self.spatial, key):
                        setattr(self.spatial, key, value)
            
            if "dynamics" in preset:
                for key, value in preset["dynamics"].items():
                    if hasattr(self.dynamics, key):
                        setattr(self.dynamics, key, value)
            
            print(f"Loaded preset: {preset_name}")
            return True
    
    def save_current_as_preset(self, preset_name: str):
        """将当前参数保存为预设"""
        with self._lock:
            self.parameter_presets[preset_name] = {
                "timing": asdict(self.timing),
                "spatial": asdict(self.spatial),
                "dynamics": asdict(self.dynamics),
                "stability": asdict(self.stability),
                "trajectory": asdict(self.trajectory)
            }
            print(f"Saved current parameters as preset: {preset_name}")
    
    def set_walking_speed(self, speed: float):
        """设置行走速度并自动调整相关参数"""
        with self._lock:
            # 限制速度范围
            speed = np.clip(speed, 0.0, self.dynamics.max_walking_speed)
            self.dynamics.walking_speed = speed
            
            # 根据速度调整步长
            self.spatial.step_length = self.dynamics.calculate_step_length_from_speed(
                self.timing.gait_cycle_time
            )
            
            # 限制步长范围
            self.spatial.step_length = np.clip(
                self.spatial.step_length, 0.0, self.spatial.max_step_length
            )
            
            print(f"Walking speed set to {speed:.3f} m/s, step length: {self.spatial.step_length:.3f} m")
    
    def set_step_frequency(self, frequency: float):
        """设置步频并更新相关参数"""
        with self._lock:
            self.timing.update_from_frequency(frequency)
            print(f"Step frequency set to {frequency:.2f} Hz, cycle time: {self.timing.gait_cycle_time:.2f} s")
    
    def get_step_frequency(self) -> float:
        """获取当前步频"""
        with self._lock:
            return self.timing.step_frequency
    
    def get_step_length(self) -> float:
        """获取当前步长"""
        with self._lock:
            return self.spatial.step_length
    
    def set_gait_type(self, gait_type: GaitType):
        """设置步态类型"""
        with self._lock:
            self.current_gait_type = gait_type
            
            # 根据步态类型调整参数
            if gait_type == GaitType.STATIC_WALK:
                self.timing.double_support_ratio = 0.2
            elif gait_type == GaitType.DYNAMIC_WALK:
                self.timing.double_support_ratio = 0.1
            elif gait_type == GaitType.TROT:
                self.timing.double_support_ratio = 0.0
                self.timing.swing_ratio = 0.5
            elif gait_type == GaitType.STAND:
                self.dynamics.walking_speed = 0.0
                self.spatial.step_length = 0.0
            
            self.timing.update_from_cycle_time()
            print(f"Gait type set to: {gait_type.value}")
    
    def calculate_foot_placement(self, leg_side: str, phase: float) -> Tuple[float, float, float]:
        """
        计算足部放置位置
        
        参数:
            leg_side: "left" 或 "right"
            phase: 步态相位 (0.0-1.0)
        
        返回:
            (x, y, z) 足部位置
        """
        with self._lock:
            # 前进方向位置
            if phase < self.timing.stance_ratio:
                # 支撑相：足部相对身体向后移动
                x = -self.spatial.step_length * (phase / self.timing.stance_ratio - 0.5)
            else:
                # 摆动相：足部向前摆动
                swing_phase = (phase - self.timing.stance_ratio) / self.timing.swing_ratio
                x = self.spatial.step_length * (swing_phase - 0.5)
            
            # 横向位置
            y_offset = self.spatial.step_width / 2
            if leg_side == "left":
                y = y_offset
            else:
                y = -y_offset
            
            # 垂直位置
            if phase < self.timing.stance_ratio:
                # 支撑相：脚在地面
                z = 0.0
            else:
                # 摆动相：抬脚轨迹
                swing_phase = (phase - self.timing.stance_ratio) / self.timing.swing_ratio
                z = self.spatial.step_height * np.sin(np.pi * swing_phase)
            
            return x, y, z
    
    def get_phase_info(self, current_time: float) -> Dict:
        """获取当前时间的相位信息"""
        with self._lock:
            cycle_phase = (current_time % self.timing.gait_cycle_time) / self.timing.gait_cycle_time
            
            # 左腿和右腿相位相差0.5个周期
            left_phase = cycle_phase
            right_phase = (cycle_phase + 0.5) % 1.0
            
            # 判断各腿的状态
            left_in_swing = left_phase >= self.timing.stance_ratio
            right_in_swing = right_phase >= self.timing.stance_ratio
            
            return {
                "cycle_phase": cycle_phase,
                "left_phase": left_phase,
                "right_phase": right_phase,
                "left_in_swing": left_in_swing,
                "right_in_swing": right_in_swing,
                "double_support": not (left_in_swing or right_in_swing)
            }
    
    def export_to_file(self, filename: str, format: str = "yaml"):
        """导出参数到文件"""
        with self._lock:
            data = {
                "gait_type": self.current_gait_type.value,
                "timing": asdict(self.timing),
                "spatial": asdict(self.spatial),
                "dynamics": asdict(self.dynamics),
                "stability": asdict(self.stability),
                "trajectory": asdict(self.trajectory),
                "presets": self.parameter_presets
            }
            
            if format.lower() == "yaml":
                with open(filename, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Parameters exported to {filename}")
    
    def import_from_file(self, filename: str, format: str = "auto"):
        """从文件导入参数"""
        with self._lock:
            if format == "auto":
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    format = "yaml"
                elif filename.endswith(".json"):
                    format = "json"
                else:
                    raise ValueError("Cannot determine file format")
            
            if format.lower() == "yaml":
                with open(filename, 'r') as f:
                    data = yaml.safe_load(f)
            elif format.lower() == "json":
                with open(filename, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # 更新参数
            if "gait_type" in data:
                self.current_gait_type = GaitType(data["gait_type"])
            
            if "timing" in data:
                for key, value in data["timing"].items():
                    if hasattr(self.timing, key):
                        setattr(self.timing, key, value)
            
            if "spatial" in data:
                for key, value in data["spatial"].items():
                    if hasattr(self.spatial, key):
                        setattr(self.spatial, key, value)
            
            if "dynamics" in data:
                for key, value in data["dynamics"].items():
                    if hasattr(self.dynamics, key):
                        setattr(self.dynamics, key, value)
            
            if "stability" in data:
                for key, value in data["stability"].items():
                    if hasattr(self.stability, key):
                        setattr(self.stability, key, value)
            
            if "trajectory" in data:
                for key, value in data["trajectory"].items():
                    if hasattr(self.trajectory, key):
                        setattr(self.trajectory, key, value)
            
            if "presets" in data:
                self.parameter_presets.update(data["presets"])
            
            print(f"Parameters imported from {filename}")
    
    def print_summary(self):
        """打印参数摘要"""
        print("\n=== 步态参数摘要 ===")
        print(f"步态类型: {self.current_gait_type.value}")
        print(f"周期时间: {self.timing.gait_cycle_time:.2f}s")
        print(f"步频: {self.timing.step_frequency:.2f}Hz")
        print(f"步长: {self.spatial.step_length:.3f}m")
        print(f"步宽: {self.spatial.step_width:.3f}m")
        print(f"抬脚高度: {self.spatial.step_height:.3f}m")
        print(f"行走速度: {self.dynamics.walking_speed:.3f}m/s")
        print(f"摆动相比例: {self.timing.swing_ratio:.1%}")
        print(f"支撑相比例: {self.timing.stance_ratio:.1%}")
        print(f"双支撑相比例: {self.timing.double_support_ratio:.1%}")


# 全局步态参数管理器实例
global_gait_manager = GaitParameterManager()


def get_gait_manager() -> GaitParameterManager:
    """获取全局步态参数管理器实例"""
    return global_gait_manager


if __name__ == "__main__":
    # 测试代码
    manager = get_gait_manager()
    
    # 打印初始参数
    manager.print_summary()
    
    # 测试预设加载
    print("\n=== 测试预设加载 ===")
    manager.load_preset("fast_walk")
    manager.print_summary()
    
    # 测试速度设置
    print("\n=== 测试速度设置 ===")
    manager.set_walking_speed(0.3)
    manager.print_summary()
    
    # 测试相位计算
    print("\n=== 测试相位计算 ===")
    phase_info = manager.get_phase_info(0.6)
    print(f"相位信息: {phase_info}")
    
    # 测试足部位置计算
    print("\n=== 测试足部位置计算 ===")
    left_pos = manager.calculate_foot_placement("left", 0.7)
    right_pos = manager.calculate_foot_placement("right", 0.7)
    print(f"左脚位置: {left_pos}")
    print(f"右脚位置: {right_pos}") 