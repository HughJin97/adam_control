"""
MuJoCo Integration Example - MuJoCo仿真集成示例
展示如何在MuJoCo仿真中使用传感器读取模块和数据总线

作者: Adam Control Team
版本: 1.0
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from sensor_reader import create_sensor_reader, read_sensors
from control_sender import create_control_sender, send_commands, ControlMode
from data_bus import get_data_bus, ContactState


class MuJoCoSimulation:
    """MuJoCo仿真环境封装"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """
        初始化仿真环境
        
        参数:
            model_path: MuJoCo模型文件路径
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 创建传感器读取器
        self.sensor_reader = create_sensor_reader(self.model, self.data)
        
        # 创建控制发送器
        self.control_sender = create_control_sender(self.model, self.data)
        
        # 获取数据总线引用
        self.data_bus = get_data_bus()
        
        # 控制参数
        self.control_frequency = 1000  # Hz
        self.control_dt = 1.0 / self.control_frequency
        
        print(f"MuJoCo simulation initialized with model: {model_path}")
        print(f"Control frequency: {self.control_frequency} Hz")
    
    def reset(self):
        """重置仿真到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始关节位置（可选）
        self._set_initial_pose()
        
        # 前向运动学
        mujoco.mj_forward(self.model, self.data)
        
        # 读取初始传感器数据
        read_sensors(self.sensor_reader)
        
        # 重置控制指令
        self.control_sender.reset_commands()
        
        print("Simulation reset to initial state")
    
    def _set_initial_pose(self):
        """设置机器人初始姿势"""
        # 设置基座高度
        if self.model.nq > 2:
            self.data.qpos[2] = 1.0  # z高度设为1米
        
        # 设置一些关节的初始角度
        joint_initial_positions = {
            "J_hip_l_roll": 0.0,
            "J_hip_l_pitch": -0.2,
            "J_knee_l_pitch": 0.4,
            "J_ankle_l_pitch": -0.2,
            "J_hip_r_roll": 0.0,
            "J_hip_r_pitch": -0.2,
            "J_knee_r_pitch": 0.4,
            "J_ankle_r_pitch": -0.2,
        }
        
        for joint_name, position in joint_initial_positions.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_idx] = position
            except:
                pass
    
    def step(self):
        """执行一个仿真步"""
        # 1. 读取传感器数据
        success = read_sensors(self.sensor_reader)
        if not success:
            print("Warning: Failed to read sensors")
        
        # 2. 执行控制计算（这里应该调用控制器）
        self._compute_control()
        
        # 3. 发送控制指令
        success = send_commands(self.control_sender)
        if not success:
            print("Warning: Failed to send commands")
        
        # 4. 执行仿真步
        mujoco.mj_step(self.model, self.data)
    
    def _compute_control(self):
        """
        计算控制输出（示例）
        在实际应用中，这里应该调用真正的控制器
        """
        # 获取IMU数据
        imu_data = self.data_bus.imu
        roll = imu_data.roll
        pitch = imu_data.pitch
        
        # 获取足部接触状态
        left_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        
        # 根据控制模式设置控制输出
        if self.control_sender.control_mode == ControlMode.TORQUE:
            # 力矩控制模式
            self._compute_torque_control()
        elif self.control_sender.control_mode == ControlMode.POSITION:
            # 位置控制模式
            self._compute_position_control()
        elif self.control_sender.control_mode == ControlMode.HYBRID:
            # 混合控制模式
            self._compute_hybrid_control()
    
    def _compute_torque_control(self):
        """计算力矩控制输出"""
        # 简单的平衡控制示例
        kp_balance = 100.0
        kd_balance = 10.0
        
        imu_data = self.data_bus.imu
        
        for joint_name in self.data_bus.joint_names:
            # 获取当前状态
            position = self.data_bus.get_joint_position(joint_name)
            velocity = self.data_bus.get_joint_velocity(joint_name)
            
            if position is None or velocity is None:
                continue
            
            # 计算期望位置（保持初始姿势）
            desired_position = 0.0
            if "knee" in joint_name:
                desired_position = 0.4
            elif "hip_pitch" in joint_name:
                desired_position = -0.2
            elif "ankle_pitch" in joint_name:
                desired_position = -0.2
            
            # 添加平衡补偿
            if "hip_roll" in joint_name:
                # 使用IMU的roll角进行平衡
                desired_position -= kp_balance * imu_data.roll * 0.1
            elif "ankle_roll" in joint_name:
                # 踝关节也参与平衡
                desired_position -= kp_balance * imu_data.roll * 0.05
            
            # PD控制
            kp = 50.0
            kd = 5.0
            torque = kp * (desired_position - position) - kd * velocity
            
            # 设置控制力矩到数据总线
            self.data_bus.set_control_torque(joint_name, torque)
    
    def _compute_position_control(self):
        """计算位置控制输出"""
        # 位置控制示例
        time_now = self.data.time
        
        for joint_name in self.data_bus.joint_names:
            # 设置期望位置
            desired_position = 0.0
            
            if "knee" in joint_name:
                # 膝关节做周期性运动
                desired_position = 0.4 + 0.2 * np.sin(2 * np.pi * 0.5 * time_now)
            elif "hip_pitch" in joint_name:
                desired_position = -0.2
            elif "ankle_pitch" in joint_name:
                desired_position = -0.2
            elif "hip_roll" in joint_name:
                # 髋关节横滚做小幅摆动
                desired_position = 0.1 * np.sin(2 * np.pi * 0.2 * time_now)
            
            # 设置控制位置到数据总线
            self.data_bus.set_control_position(joint_name, desired_position)
    
    def _compute_hybrid_control(self):
        """计算混合控制输出"""
        # 混合控制示例：腿部用位置控制，手臂用力矩控制
        time_now = self.data.time
        
        for joint_name in self.data_bus.joint_names:
            if "arm" in joint_name:
                # 手臂用力矩控制
                self.control_sender.set_joint_control_mode(joint_name, ControlMode.TORQUE)
                
                # 简单的重力补偿
                torque = 0.0
                if "arm_l_02" in joint_name or "arm_r_02" in joint_name:
                    torque = -5.0  # 抵抗重力
                
                self.data_bus.set_control_torque(joint_name, torque)
                
            else:
                # 腿部用位置控制
                self.control_sender.set_joint_control_mode(joint_name, ControlMode.POSITION)
                
                desired_position = 0.0
                if "knee" in joint_name:
                    desired_position = 0.4 + 0.1 * np.sin(2 * np.pi * 0.5 * time_now)
                elif "hip_pitch" in joint_name:
                    desired_position = -0.2
                elif "ankle_pitch" in joint_name:
                    desired_position = -0.2
                
                self.data_bus.set_control_position(joint_name, desired_position)
    
    def set_control_mode(self, mode: ControlMode):
        """设置控制模式"""
        self.control_sender.set_control_mode(mode)
        print(f"Simulation control mode set to: {mode.value}")
    
    def run_visualization(self, duration: float = 10.0):
        """
        运行带可视化的仿真
        
        参数:
            duration: 仿真时长（秒）
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 重置仿真
            self.reset()
            
            start_time = time.time()
            sim_time = 0.0
            
            while viewer.is_running() and sim_time < duration:
                step_start = time.time()
                
                # 执行仿真步
                self.step()
                
                # 更新可视化
                viewer.sync()
                
                # 控制仿真速度
                elapsed = time.time() - step_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
                
                sim_time = time.time() - start_time
                
                # 每秒打印一次状态
                if int(sim_time) > int(sim_time - self.control_dt):
                    self._print_status()
    
    def run_headless(self, duration: float = 10.0, print_interval: float = 1.0):
        """
        无可视化运行仿真
        
        参数:
            duration: 仿真时长（秒）
            print_interval: 状态打印间隔（秒）
        """
        # 重置仿真
        self.reset()
        
        start_time = time.time()
        last_print_time = start_time
        step_count = 0
        
        while (time.time() - start_time) < duration:
            # 执行仿真步
            self.step()
            step_count += 1
            
            # 定期打印状态
            current_time = time.time()
            if (current_time - last_print_time) >= print_interval:
                self._print_status()
                print(f"Steps: {step_count}, Sim time: {self.data.time:.2f}s")
                last_print_time = current_time
        
        print(f"\nSimulation completed: {step_count} steps in {duration:.2f} seconds")
        print(f"Average frequency: {step_count/duration:.1f} Hz")
    
    def _print_status(self):
        """打印当前机器人状态"""
        print("\n" + "="*50)
        print("Robot Status:")
        print("="*50)
        
        # IMU状态
        imu = self.data_bus.imu
        print(f"IMU - Roll: {np.degrees(imu.roll):.1f}°, "
              f"Pitch: {np.degrees(imu.pitch):.1f}°, "
              f"Yaw: {np.degrees(imu.yaw):.1f}°")
        
        # 质心状态
        com = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        print(f"CoM - Position: ({com.x:.3f}, {com.y:.3f}, {com.z:.3f}) m")
        print(f"CoM - Velocity: ({com_vel.x:.3f}, {com_vel.y:.3f}, {com_vel.z:.3f}) m/s")
        
        # 足部接触状态
        left_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        left_force = self.data_bus.end_effectors["left_foot"].contact_force_magnitude
        right_force = self.data_bus.end_effectors["right_foot"].contact_force_magnitude
        
        print(f"Left foot - Contact: {left_contact.name}, Force: {left_force:.1f} N")
        print(f"Right foot - Contact: {right_contact.name}, Force: {right_force:.1f} N")
        
        # 控制模式和执行器信息
        print(f"Control mode: {self.control_sender.control_mode.value}")
        
        # 显示一些关节的控制信息
        sample_joints = ["J_hip_l_roll", "J_knee_l_pitch", "J_arm_r_02"]
        actuator_info = self.control_sender.get_actuator_info()
        
        print("\nSample joint controls:")
        for joint in sample_joints:
            if joint in actuator_info:
                info = actuator_info[joint]
                control_value = info['current_control']
                mode = info['control_mode']
                print(f"  {joint}: {control_value:.2f} ({mode})")
        
        # 关节限位状态
        limits_status = self.sensor_reader.get_joint_limits_status()
        joints_near_limit = []
        for joint_name, status in limits_status.items():
            if status["near_lower_limit"] or status["near_upper_limit"]:
                joints_near_limit.append(joint_name)
        
        if joints_near_limit:
            print(f"Joints near limits: {', '.join(joints_near_limit)}")
        
        # 系统模式
        print(f"Robot mode: {self.data_bus.get_robot_mode()}")
        print("="*50)


def demonstrate_control_modes():
    """演示不同的控制模式"""
    print("\n" + "="*50)
    print("Demonstrating Different Control Modes")
    print("="*50)
    
    # 创建仿真环境
    sim = MuJoCoSimulation("models/scene.xml")
    
    # 设置机器人模式
    sim.data_bus.set_robot_mode("DEMONSTRATION")
    
    # 1. 力矩控制模式
    print("\n1. TORQUE CONTROL MODE")
    print("-" * 30)
    sim.set_control_mode(ControlMode.TORQUE)
    sim.run_headless(duration=3.0, print_interval=1.5)
    
    # 2. 位置控制模式
    print("\n2. POSITION CONTROL MODE")
    print("-" * 30)
    sim.set_control_mode(ControlMode.POSITION)
    sim.run_headless(duration=3.0, print_interval=1.5)
    
    # 3. 混合控制模式
    print("\n3. HYBRID CONTROL MODE")
    print("-" * 30)
    sim.set_control_mode(ControlMode.HYBRID)
    sim.run_headless(duration=3.0, print_interval=1.5)
    
    print("\nControl mode demonstration completed!")


def main():
    """主函数"""
    print("MuJoCo Integration Example with Control Sender")
    print("-" * 50)
    
    # 创建仿真环境
    sim = MuJoCoSimulation("models/scene.xml")
    
    # 设置机器人模式
    sim.data_bus.set_robot_mode("STANDING")
    
    # 设置控制模式
    control_mode = ControlMode.POSITION  # 可以改为 TORQUE, VELOCITY, HYBRID
    sim.set_control_mode(control_mode)
    
    # 选择运行模式
    use_visualization = True
    demonstrate_modes = False
    
    if demonstrate_modes:
        # 演示不同控制模式
        demonstrate_control_modes()
    elif use_visualization:
        print(f"\nRunning simulation with visualization in {control_mode.value} mode...")
        sim.run_visualization(duration=20.0)
    else:
        print(f"\nRunning headless simulation in {control_mode.value} mode...")
        sim.run_headless(duration=10.0, print_interval=1.0)
    
    # 打印最终统计
    print("\nFinal statistics:")
    sim.data_bus.print_status()
    
    # 获取最终的执行器信息
    actuator_info = sim.control_sender.get_actuator_info()
    print("\nFinal actuator states (sample):")
    for i, (joint_name, info) in enumerate(actuator_info.items()):
        if i < 5:  # 只显示前5个
            print(f"{joint_name}: control={info['current_control']:.3f}, mode={info['control_mode']}")


if __name__ == "__main__":
    main() 