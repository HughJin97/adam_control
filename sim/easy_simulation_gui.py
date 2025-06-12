#!/usr/bin/env python3
"""
Easy Simulation (GUI) - GUI版本仿真调试脚本

专门用于mjpython运行，提供可视化界面用于观察机器人行为：
1. 实时观察机器人姿态变化
2. 验证数据总线与仿真GUI的一致性
3. 测试不同控制模式

运行方法:
mjpython easy_simulation_gui.py

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

# 导入控制模块
from data_bus import get_data_bus, Vector3D, ContactState
from sensor_reader import SensorReader
from control_sender import ControlSender, ControlMode


class GUISimulation:
    """GUI版本仿真调试类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化仿真"""
        self.model_path = model_path
        self.model = None
        self.data = None
        
        # 控制组件
        self.sensor_reader = None
        self.control_sender = None
        self.data_bus = get_data_bus()
        
        # 仿真参数
        self.sim_time = 0.0
        self.sim_step_time = 0.001  # 1ms 仿真步长
        
        # 调试参数
        self.print_interval = 2.0  # 每2秒打印一次
        self.last_print_time = 0.0
        self.debug_joints = ["J_hip_l_pitch", "J_hip_r_pitch", "J_knee_l_pitch", "J_knee_r_pitch"]
        
        print("GUISimulation initialized")
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
        # 设置一个稳定的站立姿态
        initial_joint_positions = {
            # 左腿
            "J_hip_l_pitch": -0.05,   # 髋关节稍微前倾
            "J_knee_l_pitch": 0.1,    # 膝关节稍微弯曲
            "J_ankle_l_pitch": -0.05, # 踝关节补偿
            
            # 右腿
            "J_hip_r_pitch": -0.05,
            "J_knee_r_pitch": 0.1,
            "J_ankle_r_pitch": -0.05,
            
            # 手臂
            "J_arm_l_02": 0.1,
            "J_arm_r_02": -0.1,
        }
        
        # 应用初始位置
        for joint_name, position in initial_joint_positions.items():
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_idx = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_idx] = position
            except:
                pass
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        print("✓ 初始姿态设置完成")
    
    def _setup_simulation(self):
        """设置仿真组件"""
        self.sensor_reader = SensorReader(self.model, self.data)
        self.control_sender = ControlSender(self.model, self.data)
        self.control_sender.set_control_mode(ControlMode.TORQUE)
        print("✓ 仿真组件设置完成")
    
    def run_interactive_simulation(self):
        """运行交互式仿真"""
        print(f"\n=== 开始交互式仿真 ===")
        print("操作说明:")
        print("- 鼠标拖拽: 旋转视角")
        print("- 滚轮: 缩放")
        print("- 右键菜单: 查看传感器数据")
        print("- 键盘 '0': 零力矩模式")
        print("- 键盘 '1': 位置保持模式")
        print("- 键盘 '2': 重力补偿模式")
        print("- 键盘 'r': 重置姿态")
        print("- ESC 或关闭窗口: 退出")
        
        control_mode = "zero"  # 默认零力矩模式
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 打印初始状态
                self._print_system_status()
                
                while viewer.is_running():
                    step_start_time = time.time()
                    
                    # 检查键盘输入（简单实现）
                    # 注意：实际的键盘检测需要更复杂的实现
                    
                    # 1. 读取传感器数据
                    self.sensor_reader.read_sensors()
                    
                    # 2. 计算控制指令
                    self._compute_control(control_mode)
                    
                    # 3. 发送控制指令
                    self.control_sender.send_commands()
                    
                    # 4. 仿真步进
                    mujoco.mj_step(self.model, self.data)
                    self.sim_time += self.sim_step_time
                    
                    # 5. 更新显示
                    viewer.sync()
                    
                    # 6. 定期打印调试信息
                    if self.sim_time - self.last_print_time >= self.print_interval:
                        self._print_debug_info(control_mode)
                        self.last_print_time = self.sim_time
                    
                    # 7. 时间同步
                    elapsed_time = time.time() - step_start_time
                    sleep_time = self.sim_step_time - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n仿真被用户中断")
        except Exception as e:
            print(f"仿真错误: {e}")
        
        print("=== 仿真结束 ===")
    
    def _compute_control(self, control_mode: str):
        """计算控制指令"""
        if control_mode == "zero":
            # 零力矩控制
            for joint_name in self.data_bus.joint_names:
                self.data_bus.set_control_torque(joint_name, 0.0)
        
        elif control_mode == "position_hold":
            # 位置保持控制
            kp = 50.0  # 比例增益
            kd = 5.0   # 微分增益
            
            for joint_name in self.data_bus.joint_names:
                current_pos = self.data_bus.get_joint_position(joint_name) or 0.0
                current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                
                # 目标位置设为初始位置或零位置
                if joint_name in ["J_hip_l_pitch", "J_hip_r_pitch"]:
                    target_pos = -0.05
                elif joint_name in ["J_knee_l_pitch", "J_knee_r_pitch"]:
                    target_pos = 0.1
                elif joint_name in ["J_ankle_l_pitch", "J_ankle_r_pitch"]:
                    target_pos = -0.05
                else:
                    target_pos = 0.0
                
                # PD控制
                pos_error = target_pos - current_pos
                vel_error = 0.0 - current_vel
                control_torque = kp * pos_error + kd * vel_error
                
                # 限制力矩
                control_torque = np.clip(control_torque, -100.0, 100.0)
                self.data_bus.set_control_torque(joint_name, control_torque)
        
        elif control_mode == "gravity_comp":
            # 重力补偿控制
            self.data.qacc[:] = 0.0
            mujoco.mj_inverse(self.model, self.data)
            
            for joint_name in self.data_bus.joint_names:
                if joint_name in self.control_sender._joint_to_actuator:
                    actuator_id = self.control_sender._joint_to_actuator[joint_name]['actuator_id']
                    if actuator_id < len(self.data.qfrc_inverse):
                        gravity_torque = self.data.qfrc_inverse[actuator_id]
                        # 添加一些阻尼
                        current_vel = self.data_bus.get_joint_velocity(joint_name) or 0.0
                        damping_torque = -2.0 * current_vel
                        total_torque = gravity_torque + damping_torque
                        self.data_bus.set_control_torque(joint_name, total_torque)
    
    def _print_system_status(self):
        """打印系统状态"""
        print(f"\n=== 系统状态 ===")
        
        # 读取传感器数据
        self.sensor_reader.read_sensors()
        
        # 关节状态
        print("关键关节状态:")
        for joint_name in self.debug_joints:
            mujoco_pos = self._get_mujoco_joint_position(joint_name)
            databus_pos = self.data_bus.get_joint_position(joint_name)
            print(f"  {joint_name:15s}: MuJoCo={mujoco_pos:7.3f}, DataBus={databus_pos:7.3f}")
        
        # IMU数据
        imu_ori = self.data_bus.get_imu_orientation() 
        print(f"IMU姿态: quat({imu_ori.w:.3f}, {imu_ori.x:.3f}, {imu_ori.y:.3f}, {imu_ori.z:.3f})")
        
        # 质心数据
        com_pos = self.data_bus.get_center_of_mass_position()
        print(f"质心位置: ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
        
        # 接触状态
        left_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        print(f"足部接触: 左={left_contact.name if left_contact else 'None'}, 右={right_contact.name if right_contact else 'None'}")
    
    def _print_debug_info(self, control_mode):
        """打印调试信息"""
        print(f"\n=== 仿真状态 [t={self.sim_time:.1f}s, mode={control_mode}] ===")
        
        # 数据一致性检查
        max_error = 0.0
        for joint_name in self.debug_joints:
            mujoco_pos = self._get_mujoco_joint_position(joint_name)
            databus_pos = self.data_bus.get_joint_position(joint_name)
            if databus_pos is not None:
                error = abs(mujoco_pos - databus_pos)
                max_error = max(max_error, error)
        
        if max_error < 0.001:
            print("✓ 数据总线与MuJoCo数据一致")
        else:
            print(f"✗ 数据总线误差: {max_error:.6f}")
        
        # 机器人状态
        com_pos = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        print(f"质心高度: {com_pos.z:.3f}m, 速度: {np.linalg.norm([com_vel.x, com_vel.y, com_vel.z]):.3f}m/s")
        
        # 控制状态
        control_torques = self.data_bus.get_all_control_torques()
        active_torques = [torque for torque in control_torques.values() if abs(torque) > 0.1]
        print(f"活跃控制力矩数量: {len(active_torques)}")
        
        if control_mode == "zero":
            print("提示: 机器人应该在重力作用下倒塌")
        elif control_mode == "position_hold":
            print("提示: 机器人应该保持稳定姿态")
        elif control_mode == "gravity_comp":
            print("提示: 机器人应该悬浮不动")
    
    def _get_mujoco_joint_position(self, joint_name: str) -> float:
        """直接从MuJoCo获取关节位置"""
        try:
            joint_id = self.model.joint(joint_name).id
            qpos_idx = self.model.jnt_qposadr[joint_id]
            return self.data.qpos[qpos_idx]
        except:
            return 0.0


def main():
    """主函数"""
    print("=== Easy Simulation (GUI) 调试工具 ===")
    print("注意: 请使用 mjpython 运行此脚本")
    print("命令: mjpython easy_simulation_gui.py\n")
    
    try:
        # 创建仿真实例
        sim = GUISimulation()
        
        # 运行交互式仿真
        sim.run_interactive_simulation()
        
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 仿真结束 ===")


if __name__ == "__main__":
    main() 