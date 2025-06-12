"""
主控制循环 - 机器人控制系统的核心循环

功能：
1. 传感器数据读取 -> 数据总线更新
2. 控制算法计算 -> 生成控制指令  
3. 控制指令发送 -> 作用于仿真/实体机器人
4. 支持多种控制频率和控制模式
5. 性能监控和状态报告

作者: Adam Control Team
版本: 1.0
"""

import time
import threading
import numpy as np
from typing import Optional, Dict, Callable
from enum import Enum
import logging

# 导入数据总线和控制模块
from data_bus import get_data_bus, DataBus, Vector3D, ContactState, GaitPhase
from sensor_reader import SensorReader  
from control_sender import ControlSender, ControlMode


class ControlLoopMode(Enum):
    """控制循环模式"""
    IDLE = "IDLE"                    # 空闲模式 - 发送零力矩
    POSITION_HOLD = "POSITION_HOLD"  # 位置保持模式
    BALANCE = "BALANCE"              # 平衡控制模式
    WALKING = "WALKING"              # 行走控制模式
    MANUAL = "MANUAL"                # 手动控制模式


class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loop_times = []
        self.read_times = []
        self.compute_times = []
        self.write_times = []
    
    def add_timing(self, loop_time: float, read_time: float, 
                   compute_time: float, write_time: float):
        """添加时间测量数据"""
        self.loop_times.append(loop_time)
        self.read_times.append(read_time)
        self.compute_times.append(compute_time)
        self.write_times.append(write_time)
        
        # 保持窗口大小
        if len(self.loop_times) > self.window_size:
            self.loop_times.pop(0)
            self.read_times.pop(0)
            self.compute_times.pop(0)
            self.write_times.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.loop_times:
            return {}
        
        return {
            "avg_loop_time": np.mean(self.loop_times),
            "max_loop_time": np.max(self.loop_times),
            "avg_read_time": np.mean(self.read_times),
            "avg_compute_time": np.mean(self.compute_times),
            "avg_write_time": np.mean(self.write_times),
            "loop_frequency": 1.0 / np.mean(self.loop_times) if np.mean(self.loop_times) > 0 else 0
        }


class ControlLoop:
    """
    主控制循环类
    
    实现机器人控制的核心循环：
    读取传感器 -> 控制计算 -> 发送控制指令
    """
    
    def __init__(self, 
                 mujoco_model=None, 
                 mujoco_data=None,
                 control_frequency: float = 1000.0,
                 enable_monitoring: bool = True):
        """
        初始化控制循环
        
        Args:
            mujoco_model: MuJoCo模型
            mujoco_data: MuJoCo数据
            control_frequency: 控制频率 [Hz]
            enable_monitoring: 是否启用性能监控
        """
        # 基本设置
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.enable_monitoring = enable_monitoring
        
        # 获取数据总线
        self.data_bus = get_data_bus()
        
        # 初始化传感器读取器和控制发送器
        self.sensor_reader = None
        self.control_sender = None
        
        if mujoco_model is not None and mujoco_data is not None:
            self.sensor_reader = SensorReader(mujoco_model, mujoco_data)
            self.control_sender = ControlSender(mujoco_model, mujoco_data)
        
        # 控制模式和状态
        self.current_mode = ControlLoopMode.IDLE
        self.is_running = False
        self.loop_count = 0
        
        # 性能监控
        if enable_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        # 控制线程
        self.control_thread = None
        self.stop_event = threading.Event()
        
        # 控制参数
        self.position_hold_targets = {}  # 位置保持目标
        self.balance_gains = {
            "kp_pos": 100.0,    # 位置比例增益
            "kd_pos": 10.0,     # 位置微分增益
            "kp_ori": 50.0,     # 姿态比例增益
            "kd_ori": 5.0       # 姿态微分增益
        }
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"ControlLoop initialized - Frequency: {control_frequency} Hz")
    
    def set_mujoco_interface(self, mujoco_model, mujoco_data):
        """设置MuJoCo接口"""
        self.sensor_reader = SensorReader(mujoco_model, mujoco_data)
        self.control_sender = ControlSender(mujoco_model, mujoco_data)
        print("MuJoCo interface updated")
    
    def set_control_mode(self, mode: ControlLoopMode):
        """设置控制模式"""
        self.current_mode = mode
        self.data_bus.set_robot_mode(mode.value)
        print(f"Control mode set to: {mode.value}")
        
        # 模式切换时的初始化
        if mode == ControlLoopMode.POSITION_HOLD:
            self._initialize_position_hold()
        elif mode == ControlLoopMode.BALANCE:
            self._initialize_balance_control()
    
    def _initialize_position_hold(self):
        """初始化位置保持模式"""
        # 记录当前位置作为保持目标
        current_positions = self.data_bus.get_all_joint_positions()
        self.position_hold_targets = current_positions.copy()
        print("Position hold targets set to current joint positions")
    
    def _initialize_balance_control(self):
        """初始化平衡控制模式"""
        # 设置平衡控制的初始参数
        self.data_bus.set_gait_phase(GaitPhase.DOUBLE_SUPPORT)
        print("Balance control initialized")
    
    def read_sensors(self):
        """
        传感器读取阶段
        从仿真环境读取传感器数据并更新数据总线
        """
        if self.sensor_reader is None:
            return
        
        try:
            # 读取所有传感器数据
            self.sensor_reader.read_sensors()
            
        except Exception as e:
            self.logger.error(f"Sensor reading error: {e}")
    
    def compute_control(self):
        """
        控制计算阶段
        根据当前模式计算控制指令
        """
        try:
            if self.current_mode == ControlLoopMode.IDLE:
                self._compute_idle_control()
            elif self.current_mode == ControlLoopMode.POSITION_HOLD:
                self._compute_position_hold_control()
            elif self.current_mode == ControlLoopMode.BALANCE:
                self._compute_balance_control()
            elif self.current_mode == ControlLoopMode.WALKING:
                self._compute_walking_control()
            elif self.current_mode == ControlLoopMode.MANUAL:
                self._compute_manual_control()
                
        except Exception as e:
            self.logger.error(f"Control computation error: {e}")
            # 错误时切换到空闲模式
            self._compute_idle_control()
    
    def _compute_idle_control(self):
        """空闲模式控制 - 发送零力矩"""
        for joint_name in self.data_bus.joint_names:
            self.data_bus.set_control_torque(joint_name, 0.0)
    
    def _compute_position_hold_control(self):
        """位置保持控制 - PD控制保持目标位置"""
        kp = 100.0  # 比例增益
        kd = 10.0   # 微分增益
        
        for joint_name in self.data_bus.joint_names:
            if joint_name in self.position_hold_targets:
                # 当前状态
                current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
                current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                
                # 目标状态
                target_pos = self.position_hold_targets[joint_name]
                target_vel = 0.0
                
                # PD控制
                position_error = target_pos - current_pos
                velocity_error = target_vel - current_vel
                
                control_torque = kp * position_error + kd * velocity_error
                
                # 限制力矩范围
                control_torque = np.clip(control_torque, -100.0, 100.0)
                
                self.data_bus.set_control_torque(joint_name, control_torque)
    
    def _compute_balance_control(self):
        """平衡控制 - 基于IMU的简单平衡控制"""
        # 获取IMU数据
        imu_orientation = self.data_bus.get_imu_orientation()
        imu_angular_vel = self.data_bus.get_imu_angular_velocity()
        
        # 简化的平衡控制 - 主要控制腿部关节
        leg_joints = [
            "J_hip_l_pitch", "J_hip_r_pitch",
            "J_knee_l_pitch", "J_knee_r_pitch", 
            "J_ankle_l_pitch", "J_ankle_r_pitch"
        ]
        
        # 基于IMU俯仰角的反馈控制
        pitch_angle = np.arcsin(2 * (imu_orientation.w * imu_orientation.y - 
                                   imu_orientation.z * imu_orientation.x))
        pitch_rate = imu_angular_vel.y
        
        # PD控制参数
        kp_balance = 200.0
        kd_balance = 20.0
        
        balance_torque = -(kp_balance * pitch_angle + kd_balance * pitch_rate)
        
        # 分配给腿部关节
        for joint_name in leg_joints:
            if "hip" in joint_name and "pitch" in joint_name:
                self.data_bus.set_control_torque(joint_name, balance_torque * 0.5)
            elif "ankle" in joint_name and "pitch" in joint_name:
                self.data_bus.set_control_torque(joint_name, balance_torque * 0.3)
            else:
                self.data_bus.set_control_torque(joint_name, 0.0)
        
        # 其他关节保持当前位置
        other_joints = [j for j in self.data_bus.joint_names if j not in leg_joints]
        for joint_name in other_joints:
            current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
            current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
            hold_torque = 50.0 * (0.0 - current_pos) + 5.0 * (0.0 - current_vel)
            self.data_bus.set_control_torque(joint_name, hold_torque)
    
    def _compute_walking_control(self):
        """行走控制 - 占位符实现"""
        # TODO: 实现行走控制算法
        # 当前暂时使用位置保持
        self._compute_position_hold_control()
    
    def _compute_manual_control(self):
        """手动控制模式 - 直接使用数据总线中的设定值"""
        # 手动模式下，控制指令由外部设定，这里不做计算
        pass
    
    def send_control(self):
        """
        控制发送阶段
        将计算得到的控制指令发送到仿真环境
        """
        if self.control_sender is None:
            return
        
        try:
            # 发送控制指令
            self.control_sender.send_commands()
            
        except Exception as e:
            self.logger.error(f"Control sending error: {e}")
    
    def run_single_loop(self):
        """执行单次控制循环"""
        loop_start_time = time.time()
        
        # 1. 读取传感器数据
        read_start_time = time.time()
        self.read_sensors()
        read_time = time.time() - read_start_time
        
        # 2. 计算控制指令
        compute_start_time = time.time()
        self.compute_control()
        compute_time = time.time() - compute_start_time
        
        # 3. 发送控制指令
        write_start_time = time.time() 
        self.send_control()
        write_time = time.time() - write_start_time
        
        loop_time = time.time() - loop_start_time
        
        # 性能监控
        if self.enable_monitoring:
            self.performance_monitor.add_timing(loop_time, read_time, compute_time, write_time)
        
        self.loop_count += 1
        
        return loop_time
    
    def _control_thread_function(self):
        """控制线程主函数"""
        print(f"Control thread started - Target frequency: {self.control_frequency} Hz")
        
        while not self.stop_event.is_set():
            loop_start_time = time.time()
            
            # 执行控制循环
            self.run_single_loop()
            
            # 计算睡眠时间以维持控制频率
            loop_duration = time.time() - loop_start_time
            sleep_time = self.control_period - loop_duration
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_duration > self.control_period * 1.1:  # 超时10%以上才警告
                self.logger.warning(f"Control loop overtime: {loop_duration:.4f}s > {self.control_period:.4f}s")
        
        print("Control thread stopped")
    
    def start(self):
        """启动控制循环"""
        if self.is_running:
            print("Control loop is already running")
            return
        
        if self.sensor_reader is None or self.control_sender is None:
            print("Error: MuJoCo interface not set. Call set_mujoco_interface() first.")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.loop_count = 0
        
        # 启动控制线程
        self.control_thread = threading.Thread(target=self._control_thread_function)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("Control loop started")
    
    def stop(self):
        """停止控制循环"""
        if not self.is_running:
            print("Control loop is not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # 等待控制线程结束
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        print("Control loop stopped")
    
    def get_status(self) -> Dict:
        """获取控制循环状态"""
        status = {
            "is_running": self.is_running,
            "current_mode": self.current_mode.value,
            "control_frequency": self.control_frequency,
            "loop_count": self.loop_count
        }
        
        # 添加性能统计
        if self.enable_monitoring and hasattr(self, 'performance_monitor'):
            perf_stats = self.performance_monitor.get_stats()
            status.update(perf_stats)
        
        return status
    
    def print_status(self):
        """打印控制循环状态"""
        status = self.get_status()
        print("=" * 60)
        print("Control Loop Status")
        print("=" * 60)
        for key, value in status.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)
    
    def set_position_hold_target(self, joint_name: str, position: float):
        """设置位置保持目标"""
        if joint_name in self.data_bus.joint_names:
            self.position_hold_targets[joint_name] = position
            print(f"Position hold target set: {joint_name} = {position:.3f}")
    
    def set_balance_gains(self, kp_pos: float = None, kd_pos: float = None,
                         kp_ori: float = None, kd_ori: float = None):
        """设置平衡控制增益"""
        if kp_pos is not None:
            self.balance_gains["kp_pos"] = kp_pos
        if kd_pos is not None:
            self.balance_gains["kd_pos"] = kd_pos
        if kp_ori is not None:
            self.balance_gains["kp_ori"] = kp_ori
        if kd_ori is not None:
            self.balance_gains["kd_ori"] = kd_ori
        
        print(f"Balance gains updated: {self.balance_gains}")


# 便捷函数
def create_control_loop(mujoco_model=None, mujoco_data=None, 
                       frequency: float = 1000.0) -> ControlLoop:
    """创建控制循环实例"""
    return ControlLoop(mujoco_model, mujoco_data, frequency)


if __name__ == "__main__":
    # 测试代码 - 演示控制循环的基本用法
    print("Testing ControlLoop...")
    
    # 创建控制循环（不连接MuJoCo）
    control_loop = ControlLoop(control_frequency=100.0, enable_monitoring=True)
    
    # 测试不同控制模式
    control_loop.set_control_mode(ControlLoopMode.IDLE)
    control_loop.set_control_mode(ControlLoopMode.POSITION_HOLD)
    
    # 打印状态
    control_loop.print_status()
    
    print("ControlLoop test completed") 