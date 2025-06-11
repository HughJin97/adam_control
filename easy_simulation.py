#!/usr/bin/env python3
"""
Easy Simulation - 简单仿真调试脚本

用于在MuJoCo仿真中调试数据总线和观察机器人的基本行为：
1. 验证数据总线读写正确性
2. 对比仿真GUI与数据总线中的关节角度
3. 检查数据更新频率与仿真步长匹配
4. 观察机器人在不同控制模式下的行为

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
from typing import Dict, List

# 导入控制模块
from control_loop import ControlLoop, ControlLoopMode
from data_bus import get_data_bus, Vector3D, ContactState
from sensor_reader import SensorReader
from control_sender import ControlSender, ControlMode


class EasySimulation:
    """简单仿真调试类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化仿真"""
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        
        # 控制组件
        self.sensor_reader = None
        self.control_sender = None
        self.control_loop = None
        self.data_bus = get_data_bus()
        
        # 仿真参数
        self.sim_time = 0.0
        self.sim_step_time = 0.001  # 1ms 仿真步长
        self.control_frequency = 1000.0  # 1000Hz 控制频率
        
        # 调试参数
        self.print_interval = 1.0  # 每秒打印一次调试信息
        self.last_print_time = 0.0
        self.debug_joints = ["J_hip_l_pitch", "J_hip_r_pitch", "J_knee_l_pitch", "J_knee_r_pitch"]
        
        # 统计信息
        self.stats = {
            "sim_steps": 0,
            "control_cycles": 0,
            "sensor_reads": 0,
            "control_sends": 0,
            "start_time": 0.0
        }
        
        print("EasySimulation initialized")
        self._load_model()
        self._setup_simulation()
    
    def _load_model(self):
        """加载MuJoCo模型"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print(f"✓ 模型加载成功: {self.model_path}")
            print(f"  - DOF: {self.model.nq}")
            print(f"  - 执行器: {self.model.nu}")
            print(f"  - 传感器: {self.model.nsensor}")
            
            # 设置合理的初始姿态
            self._set_initial_pose()
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def _set_initial_pose(self):
        """设置机器人的初始姿态"""
        # 设置一个稍微弯曲膝盖的站立姿态，避免奇异位置
        initial_joint_positions = {
            # 左腿
            "J_hip_l_pitch": -0.1,   # 髋关节稍微前倾
            "J_knee_l_pitch": 0.2,   # 膝关节弯曲
            "J_ankle_l_pitch": -0.1, # 踝关节补偿
            
            # 右腿
            "J_hip_r_pitch": -0.1,
            "J_knee_r_pitch": 0.2,
            "J_ankle_r_pitch": -0.1,
            
            # 手臂放在身体两侧
            "J_arm_l_02": 0.2,  # 左肩外展
            "J_arm_r_02": -0.2, # 右肩外展
        }
        
        # 应用初始位置
        for joint_name, position in initial_joint_positions.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_idx] = position
            except:
                print(f"Warning: Could not set initial position for joint {joint_name}")
        
        # 前向动力学计算，更新状态
        mujoco.mj_forward(self.model, self.data)
        print("✓ 初始姿态设置完成")
    
    def _setup_simulation(self):
        """设置仿真组件"""
        # 创建传感器读取器和控制发送器
        self.sensor_reader = SensorReader(self.model, self.data)
        self.control_sender = ControlSender(self.model, self.data)
        
        # 设置控制发送器为力矩控制模式
        self.control_sender.set_control_mode(ControlMode.TORQUE)
        
        print("✓ 仿真组件设置完成")
    
    def run_debug_simulation(self, duration: float = 30.0, control_mode: str = "zero"):
        """
        运行调试仿真
        
        参数:
            duration: 仿真时长 (秒)
            control_mode: 控制模式 ("zero", "gravity_comp", "position_hold")
        """
        print(f"\n=== 开始调试仿真 ===")
        print(f"仿真时长: {duration}s")
        print(f"控制模式: {control_mode}")
        print(f"仿真步长: {self.sim_step_time*1000:.1f}ms")
        print(f"控制频率: {self.control_frequency}Hz")
        
        # 初始化统计
        self.stats["start_time"] = time.time()
        self.stats["sim_steps"] = 0
        self.stats["control_cycles"] = 0
        
        # 打印初始状态
        self._print_initial_state()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print("\n✓ 仿真界面已启动")
                print("提示: 在仿真界面中可以:")
                print("  - 鼠标拖拽旋转视角")
                print("  - 滚轮缩放")
                print("  - 右键菜单查看传感器数据")
                
                target_time = 0.0
                
                while viewer.is_running() and self.sim_time < duration:
                    step_start_time = time.time()
                    
                    # 1. 读取传感器数据
                    sensor_success = self.sensor_reader.read_sensors()
                    if sensor_success:
                        self.stats["sensor_reads"] += 1
                    
                    # 2. 计算控制指令
                    self._compute_control(control_mode)
                    self.stats["control_cycles"] += 1
                    
                    # 3. 发送控制指令
                    control_success = self.control_sender.send_commands()
                    if control_success:
                        self.stats["control_sends"] += 1
                    
                    # 4. 仿真步进
                    mujoco.mj_step(self.model, self.data)
                    self.sim_time += self.sim_step_time
                    self.stats["sim_steps"] += 1
                    
                    # 5. 更新显示
                    viewer.sync()
                    
                    # 6. 调试信息打印
                    if self.sim_time - self.last_print_time >= self.print_interval:
                        self._print_debug_info(control_mode)
                        self.last_print_time = self.sim_time
                    
                    # 7. 时间同步
                    target_time += self.sim_step_time
                    elapsed_time = time.time() - step_start_time
                    sleep_time = self.sim_step_time - elapsed_time
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif elapsed_time > self.sim_step_time * 2:
                        print(f"Warning: 仿真步长超时 {elapsed_time*1000:.1f}ms")
                
        except KeyboardInterrupt:
            print("\n仿真被用户中断")
        
        self._print_final_statistics()
    
    def _compute_control(self, control_mode: str):
        """计算控制指令"""
        if control_mode == "zero":
            # 零力矩控制
            for joint_name in self.data_bus.joint_names:
                self.data_bus.set_control_torque(joint_name, 0.0)
        
        elif control_mode == "gravity_comp":
            # 重力补偿控制
            mujoco.mj_inverse(self.model, self.data)
            for i, joint_name in enumerate(self.data_bus.joint_names):
                if joint_name in self.control_sender._joint_to_actuator:
                    # 使用逆动力学计算的重力补偿力矩
                    actuator_id = self.control_sender._joint_to_actuator[joint_name]['actuator_id']
                    if actuator_id < len(self.data.qfrc_inverse):
                        gravity_torque = self.data.qfrc_inverse[actuator_id]
                        self.data_bus.set_control_torque(joint_name, gravity_torque)
        
        elif control_mode == "position_hold":
            # 位置保持控制
            kp = 50.0  # 比例增益
            kd = 5.0   # 微分增益
            
            for joint_name in self.data_bus.joint_names:
                current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
                current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                
                # 目标位置（保持当前位置）
                target_pos = current_pos  # 或者设置特定的目标位置
                target_vel = 0.0
                
                # PD控制
                pos_error = target_pos - current_pos
                vel_error = target_vel - current_vel
                control_torque = kp * pos_error + kd * vel_error
                
                # 限制力矩
                control_torque = np.clip(control_torque, -50.0, 50.0)
                self.data_bus.set_control_torque(joint_name, control_torque)
    
    def _print_initial_state(self):
        """打印初始状态"""
        print(f"\n=== 初始状态 ===")
        
        # 读取一次传感器数据
        self.sensor_reader.read_sensors()
        
        # 打印关节位置
        print("关键关节初始位置:")
        for joint_name in self.debug_joints:
            mujoco_pos = self._get_mujoco_joint_position(joint_name)
            databus_pos = self.data_bus.get_joint_position(joint_name)
            print(f"  {joint_name:15s}: MuJoCo={mujoco_pos:7.3f}, DataBus={databus_pos:7.3f}")
        
        # 打印IMU数据
        imu_ori = self.data_bus.get_imu_orientation()
        print(f"IMU姿态: quat({imu_ori.w:.3f}, {imu_ori.x:.3f}, {imu_ori.y:.3f}, {imu_ori.z:.3f})")
        
        # 打印质心位置
        com_pos = self.data_bus.get_center_of_mass_position()
        print(f"质心位置: ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
        
        # 打印足部接触状态
        left_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        print(f"足部接触: 左足={left_contact.name if left_contact else 'None'}, 右足={right_contact.name if right_contact else 'None'}")
    
    def _print_debug_info(self, control_mode: str):
        """打印调试信息"""
        print(f"\n=== 仿真状态 [t={self.sim_time:.1f}s] ===")
        
        # 关节位置对比
        print("关节位置对比 (MuJoCo vs DataBus):")
        max_error = 0.0
        for joint_name in self.debug_joints:
            mujoco_pos = self._get_mujoco_joint_position(joint_name)
            databus_pos = self.data_bus.get_joint_position(joint_name)
            error = abs(mujoco_pos - databus_pos) if databus_pos is not None else float('inf')
            max_error = max(max_error, error)
            
            status = "✓" if error < 0.001 else "✗"
            print(f"  {status} {joint_name:15s}: {mujoco_pos:7.3f} vs {databus_pos:7.3f} (err={error:.6f})")
        
        # 数据一致性检查
        if max_error < 0.001:
            print("✓ 数据总线与MuJoCo数据一致")
        else:
            print(f"✗ 数据总线存在误差，最大误差: {max_error:.6f}")
        
        # IMU数据
        imu_ori = self.data_bus.get_imu_orientation()
        imu_vel = self.data_bus.get_imu_angular_velocity()
        print(f"IMU: 姿态=quat({imu_ori.w:.3f},{imu_ori.x:.3f},{imu_ori.y:.3f},{imu_ori.z:.3f})")
        print(f"     角速度=({imu_vel.x:.3f},{imu_vel.y:.3f},{imu_vel.z:.3f})")
        
        # 质心数据
        com_pos = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        print(f"质心: 位置=({com_pos.x:.3f},{com_pos.y:.3f},{com_pos.z:.3f})")
        print(f"     速度=({com_vel.x:.3f},{com_vel.y:.3f},{com_vel.z:.3f})")
        
        # 控制状态
        control_torques = self.data_bus.get_all_control_torques()
        active_torques = [f"{name}={torque:.2f}" for name, torque in control_torques.items() 
                         if abs(torque) > 0.01]
        if active_torques:
            print(f"活跃控制力矩: {', '.join(active_torques[:3])}...")
        else:
            print("控制力矩: 全部为零")
        
        # 性能统计
        elapsed_time = time.time() - self.stats["start_time"]
        if elapsed_time > 0:
            sim_fps = self.stats["sim_steps"] / elapsed_time
            control_fps = self.stats["control_cycles"] / elapsed_time
            print(f"性能: 仿真={sim_fps:.1f}Hz, 控制={control_fps:.1f}Hz")
    
    def _get_mujoco_joint_position(self, joint_name: str) -> float:
        """直接从MuJoCo获取关节位置"""
        try:
            joint_id = self.model.joint(joint_name).id
            qpos_idx = self.model.jnt_qposadr[joint_id]
            return self.data.qpos[qpos_idx]
        except:
            return 0.0
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        elapsed_time = time.time() - self.stats["start_time"]
        
        print(f"\n=== 仿真统计 ===")
        print(f"总仿真时间: {elapsed_time:.1f}s")
        print(f"仿真步数: {self.stats['sim_steps']}")
        print(f"控制周期: {self.stats['control_cycles']}")
        print(f"传感器读取: {self.stats['sensor_reads']}")
        print(f"控制发送: {self.stats['control_sends']}")
        
        if elapsed_time > 0:
            print(f"平均仿真频率: {self.stats['sim_steps']/elapsed_time:.1f} Hz")
            print(f"平均控制频率: {self.stats['control_cycles']/elapsed_time:.1f} Hz")
            
            # 成功率
            sensor_success_rate = (self.stats['sensor_reads'] / self.stats['control_cycles']) * 100
            control_success_rate = (self.stats['control_sends'] / self.stats['control_cycles']) * 100
            print(f"传感器读取成功率: {sensor_success_rate:.1f}%")
            print(f"控制发送成功率: {control_success_rate:.1f}%")
    
    def test_data_consistency(self, num_samples: int = 100):
        """测试数据总线一致性"""
        print(f"\n=== 数据一致性测试 ===")
        print(f"测试样本数: {num_samples}")
        
        errors = []
        for i in range(num_samples):
            # 随机改变关节位置
            for joint_name in self.debug_joints:
                try:
                    joint_id = self.model.joint(joint_name).id
                    qpos_idx = self.model.jnt_qposadr[joint_id]
                    self.data.qpos[qpos_idx] = np.random.uniform(-0.5, 0.5)
                except:
                    pass
            
            # 前向动力学
            mujoco.mj_forward(self.model, self.data)
            
            # 读取传感器
            self.sensor_reader.read_sensors()
            
            # 检查一致性
            sample_errors = []
            for joint_name in self.debug_joints:
                mujoco_pos = self._get_mujoco_joint_position(joint_name)
                databus_pos = self.data_bus.get_joint_position(joint_name)
                if databus_pos is not None:
                    error = abs(mujoco_pos - databus_pos)
                    sample_errors.append(error)
            
            if sample_errors:
                errors.extend(sample_errors)
        
        if errors:
            print(f"平均误差: {np.mean(errors):.8f}")
            print(f"最大误差: {np.max(errors):.8f}")
            print(f"标准差: {np.std(errors):.8f}")
            
            if np.max(errors) < 1e-6:
                print("✓ 数据一致性测试通过")
            else:
                print("✗ 数据一致性测试失败")
        else:
            print("✗ 无法获取测试数据")


def main():
    """主函数"""
    print("=== Easy Simulation 调试工具 ===\n")
    
    try:
        # 创建仿真实例
        sim = EasySimulation()
        
        # 首先进行数据一致性测试
        sim.test_data_consistency(50)
        
        print("\n选择仿真模式:")
        print("1. 零力矩控制 (观察机器人倒塌)")
        print("2. 重力补偿控制 (机器人应保持姿态)")
        print("3. 位置保持控制 (PD控制保持当前位置)")
        print("4. 自定义时长仿真")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == "1":
            sim.run_debug_simulation(duration=15.0, control_mode="zero")
        elif choice == "2":
            sim.run_debug_simulation(duration=20.0, control_mode="gravity_comp")
        elif choice == "3":
            sim.run_debug_simulation(duration=25.0, control_mode="position_hold")
        elif choice == "4":
            duration = float(input("仿真时长 (秒): "))
            mode = input("控制模式 (zero/gravity_comp/position_hold): ")
            sim.run_debug_simulation(duration=duration, control_mode=mode)
        else:
            print("无效选择，运行默认仿真")
            sim.run_debug_simulation(duration=10.0, control_mode="zero")
    
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 仿真结束 ===")


if __name__ == "__main__":
    main() 