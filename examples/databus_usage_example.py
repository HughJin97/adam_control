"""
DataBus使用示例
展示如何在机器人各个模块中使用数据总线进行数据共享

作者: Adam Control Team
版本: 1.0
"""

from data_bus import (
    get_data_bus, 
    Vector3D, 
    Quaternion, 
    ContactState, 
    GaitPhase, 
    LegState
)
import numpy as np
import time
import threading


class SensorModule:
    """传感器模块示例"""
    
    def __init__(self):
        self.data_bus = get_data_bus()
        self.running = False
        self.thread = None
    
    def start(self):
        """启动传感器数据采集"""
        self.running = True
        self.thread = threading.Thread(target=self._sensor_loop)
        self.thread.start()
        print("Sensor module started")
    
    def stop(self):
        """停止传感器数据采集"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Sensor module stopped")
    
    def _sensor_loop(self):
        """传感器数据采集循环"""
        while self.running:
            # 模拟IMU数据
            imu_orientation = Quaternion(
                w=0.707,
                x=0.0,
                y=0.0,
                z=0.707
            )
            self.data_bus.set_imu_orientation(imu_orientation)
            
            # 模拟角速度数据
            ang_vel = Vector3D(0.1, 0.0, 0.05)
            self.data_bus.set_imu_angular_velocity(ang_vel)
            
            # 模拟线加速度数据
            lin_acc = Vector3D(0.0, 0.0, -9.81)
            self.data_bus.set_imu_linear_acceleration(lin_acc)
            
            # 模拟关节传感器数据
            joint_positions = {
                "J_hip_l_roll": 0.1 + 0.05 * np.sin(time.time()),
                "J_hip_l_pitch": 0.2 + 0.1 * np.cos(time.time()),
                "J_knee_l_pitch": -0.4 + 0.1 * np.sin(time.time() * 2),
                "J_ankle_l_pitch": 0.1 + 0.05 * np.cos(time.time() * 2),
            }
            
            for joint_name, position in joint_positions.items():
                self.data_bus.set_joint_position(joint_name, position)
                # 模拟速度数据
                velocity = 0.1 * np.cos(time.time())
                self.data_bus.set_joint_velocity(joint_name, velocity)
            
            # 模拟足部接触传感器
            self.data_bus.set_end_effector_contact_state(
                "left_foot", 
                ContactState.CONTACT if np.sin(time.time()) > 0 else ContactState.NO_CONTACT
            )
            self.data_bus.set_end_effector_contact_state(
                "right_foot", 
                ContactState.CONTACT if np.sin(time.time()) < 0 else ContactState.NO_CONTACT
            )
            
            time.sleep(0.01)  # 100Hz更新频率


class GaitPlannerModule:
    """步态规划模块示例"""
    
    def __init__(self):
        self.data_bus = get_data_bus()
        self.running = False
        self.thread = None
        self.gait_cycle_time = 2.0  # 2秒一个步态周期
        self.start_time = None
    
    def start(self):
        """启动步态规划"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._gait_planning_loop)
        self.thread.start()
        print("Gait planner module started")
    
    def stop(self):
        """停止步态规划"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Gait planner module stopped")
    
    def _gait_planning_loop(self):
        """步态规划循环"""
        while self.running:
            current_time = time.time() - self.start_time
            phase_ratio = (current_time % self.gait_cycle_time) / self.gait_cycle_time
            
            # 简单的交替步态
            if phase_ratio < 0.5:
                # 左腿支撑，右腿摆动
                self.data_bus.set_leg_phase(LegState.LEFT_LEG, GaitPhase.STANCE)
                self.data_bus.set_leg_phase(LegState.RIGHT_LEG, GaitPhase.SWING)
                self.data_bus.set_gait_phase(GaitPhase.STANCE)
            else:
                # 右腿支撑，左腿摆动
                self.data_bus.set_leg_phase(LegState.LEFT_LEG, GaitPhase.SWING)
                self.data_bus.set_leg_phase(LegState.RIGHT_LEG, GaitPhase.STANCE)
                self.data_bus.set_gait_phase(GaitPhase.SWING)
            
            # 更新步态参数
            self.data_bus.gait.phase_time = phase_ratio * self.gait_cycle_time
            self.data_bus.gait.walking_speed = 0.5  # 0.5 m/s
            
            # 计算足部期望位置
            left_foot_target = Vector3D(
                x=0.0 + 0.1 * np.sin(2 * np.pi * phase_ratio),
                y=0.1,
                z=-1.0 + 0.05 * abs(np.sin(2 * np.pi * phase_ratio))
            )
            
            right_foot_target = Vector3D(
                x=0.0 + 0.1 * np.sin(2 * np.pi * phase_ratio + np.pi),
                y=-0.1,
                z=-1.0 + 0.05 * abs(np.sin(2 * np.pi * phase_ratio + np.pi))
            )
            
            self.data_bus.end_effectors["left_foot"].desired_position = left_foot_target
            self.data_bus.end_effectors["right_foot"].desired_position = right_foot_target
            
            time.sleep(0.02)  # 50Hz更新频率


class ControllerModule:
    """控制器模块示例"""
    
    def __init__(self):
        self.data_bus = get_data_bus()
        self.running = False
        self.thread = None
    
    def start(self):
        """启动控制器"""
        self.running = True
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.start()
        print("Controller module started")
    
    def stop(self):
        """停止控制器"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Controller module stopped")
    
    def _control_loop(self):
        """控制循环"""
        while self.running:
            # 获取当前关节状态
            joint_positions = self.data_bus.get_all_joint_positions()
            joint_velocities = self.data_bus.get_all_joint_velocities()
            
            # 获取步态状态
            gait_phase = self.data_bus.get_gait_phase()
            left_leg_phase = self.data_bus.get_leg_phase(LegState.LEFT_LEG)
            right_leg_phase = self.data_bus.get_leg_phase(LegState.RIGHT_LEG)
            
            # 简单的PD控制器
            kp = 100.0  # 比例增益
            kd = 10.0   # 微分增益
            
            # 为每个关节计算控制力矩
            for joint_name in self.data_bus.joint_names:
                if joint_name in joint_positions:
                    current_pos = joint_positions[joint_name]
                    current_vel = joint_velocities.get(joint_name, 0.0)
                    
                    # 简单的目标位置（这里应该来自轨迹规划）
                    if "hip" in joint_name:
                        desired_pos = 0.1 * np.sin(time.time())
                    elif "knee" in joint_name:
                        desired_pos = -0.3 + 0.1 * np.cos(time.time())
                    elif "ankle" in joint_name:
                        desired_pos = 0.0
                    else:
                        desired_pos = 0.0
                    
                    # PD控制
                    position_error = desired_pos - current_pos
                    velocity_error = 0.0 - current_vel  # 期望速度为0
                    
                    control_torque = kp * position_error + kd * velocity_error
                    
                    # 限制力矩范围
                    max_torque = 50.0
                    control_torque = np.clip(control_torque, -max_torque, max_torque)
                    
                    # 设置控制输出
                    self.data_bus.set_control_torque(joint_name, control_torque)
                    self.data_bus.set_control_position(joint_name, desired_pos)
            
            time.sleep(0.001)  # 1000Hz控制频率


class StateEstimatorModule:
    """状态估计模块示例"""
    
    def __init__(self):
        self.data_bus = get_data_bus()
        self.running = False
        self.thread = None
    
    def start(self):
        """启动状态估计"""
        self.running = True
        self.thread = threading.Thread(target=self._estimation_loop)
        self.thread.start()
        print("State estimator module started")
    
    def stop(self):
        """停止状态估计"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("State estimator module stopped")
    
    def _estimation_loop(self):
        """状态估计循环"""
        while self.running:
            # 获取IMU数据
            imu_orientation = self.data_bus.get_imu_orientation()
            imu_angular_vel = self.data_bus.get_imu_angular_velocity()
            imu_linear_acc = self.data_bus.get_imu_linear_acceleration()
            
            # 获取关节数据
            joint_positions = self.data_bus.get_all_joint_positions()
            
            # 获取足部接触状态
            left_foot_contact = self.data_bus.get_end_effector_contact_state("left_foot")
            right_foot_contact = self.data_bus.get_end_effector_contact_state("right_foot")
            
            # 简单的质心估计（这里应该使用更复杂的算法）
            estimated_com_pos = Vector3D(
                x=0.0,
                y=0.0,
                z=1.0 + 0.05 * np.sin(time.time())  # 模拟COM高度变化
            )
            
            estimated_com_vel = Vector3D(
                x=0.1 * np.cos(time.time()),
                y=0.0,
                z=0.05 * np.cos(time.time())
            )
            
            # 更新数据总线
            self.data_bus.set_center_of_mass_position(estimated_com_pos)
            self.data_bus.set_center_of_mass_velocity(estimated_com_vel)
            
            # 估计足部位置（正运动学）
            left_foot_pos = Vector3D(
                x=0.0 + 0.1 * np.sin(time.time()),
                y=0.1,
                z=-1.0 + 0.02 * np.sin(time.time() * 3)
            )
            
            right_foot_pos = Vector3D(
                x=0.0 + 0.1 * np.sin(time.time() + np.pi),
                y=-0.1,
                z=-1.0 + 0.02 * np.sin(time.time() * 3 + np.pi)
            )
            
            self.data_bus.set_end_effector_position("left_foot", left_foot_pos)
            self.data_bus.set_end_effector_position("right_foot", right_foot_pos)
            
            time.sleep(0.01)  # 100Hz更新频率


def main():
    """主函数，演示各模块协同工作"""
    print("Starting DataBus usage example...")
    
    # 创建模块实例
    sensor_module = SensorModule()
    gait_planner = GaitPlannerModule()
    controller = ControllerModule()
    state_estimator = StateEstimatorModule()
    
    try:
        # 启动所有模块
        sensor_module.start()
        time.sleep(0.1)
        
        state_estimator.start()
        time.sleep(0.1)
        
        gait_planner.start()
        time.sleep(0.1)
        
        controller.start()
        time.sleep(0.1)
        
        # 设置机器人模式
        data_bus = get_data_bus()
        data_bus.set_robot_mode("WALKING")
        
        # 运行5秒钟
        print("Running for 5 seconds...")
        for i in range(50):
            time.sleep(0.1)
            if i % 10 == 0:
                # 每秒打印一次状态
                print(f"\n--- Status at {i/10:.1f} seconds ---")
                data_bus.print_status()
                
                # 打印一些具体数据
                left_hip_pos = data_bus.get_joint_position("J_hip_l_roll")
                left_foot_contact = data_bus.get_end_effector_contact_state("left_foot")
                com_pos = data_bus.get_center_of_mass_position()
                
                print(f"Left hip roll position: {left_hip_pos:.3f} rad")
                print(f"Left foot contact: {left_foot_contact}")
                print(f"Center of mass position: ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # 停止所有模块
        controller.stop()
        gait_planner.stop()
        state_estimator.stop()
        sensor_module.stop()
        
        print("All modules stopped. Example completed.")


if __name__ == "__main__":
    main() 