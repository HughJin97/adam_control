"""
主控制循环使用示例

演示如何使用 ControlLoop 类与 MuJoCo 仿真集成，
实现完整的读取-计算-写入控制循环。

功能演示：
1. 初始化仿真环境和控制循环
2. 运行不同控制模式（空闲、位置保持、平衡控制）
3. 性能监控和状态报告
4. 简单的控制频率测试

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
from pathlib import Path

# 导入控制模块
from control_loop import ControlLoop, ControlLoopMode
from data_bus import get_data_bus


class MuJoCoControlExample:
    """MuJoCo控制循环示例类"""
    
    def __init__(self, model_path: str = "models/scene.xml"):
        """
        初始化示例
        
        Args:
            model_path: MuJoCo模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.data = None
        self.viewer = None
        self.control_loop = None
        self.data_bus = get_data_bus()
        
        # 加载MuJoCo模型
        self._load_model()
        
        # 创建控制循环
        self._create_control_loop()
        
        print("MuJoCoControlExample initialized successfully")
    
    def _load_model(self):
        """加载MuJoCo模型"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print(f"Model loaded: {self.model_path}")
            print(f"Model info: {self.model.nq} DOF, {self.model.nu} actuators")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _create_control_loop(self):
        """创建控制循环"""
        # 创建1000Hz的控制循环
        self.control_loop = ControlLoop(
            mujoco_model=self.model,
            mujoco_data=self.data,
            control_frequency=1000.0,
            enable_monitoring=True
        )
        
        print("Control loop created with 1000 Hz frequency")
    
    def run_idle_test(self, duration: float = 5.0):
        """运行空闲模式测试 - 发送零力矩"""
        print(f"\n=== Running IDLE test for {duration} seconds ===")
        
        # 设置空闲模式
        self.control_loop.set_control_mode(ControlLoopMode.IDLE)
        
        # 启动控制循环
        self.control_loop.start()
        
        # 运行指定时间
        start_time = time.time()
        while time.time() - start_time < duration:
            # 推进仿真
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.001)  # 1ms仿真步长
        
        # 停止控制循环
        self.control_loop.stop()
        
        # 打印统计信息
        self.control_loop.print_status()
        self._print_sensor_data()
    
    def run_position_hold_test(self, duration: float = 10.0):
        """运行位置保持测试"""
        print(f"\n=== Running POSITION_HOLD test for {duration} seconds ===")
        
        # 设置位置保持模式
        self.control_loop.set_control_mode(ControlLoopMode.POSITION_HOLD)
        
        # 设置一些特定的位置目标
        position_targets = {
            "J_arm_l_02": 0.5,    # 左肩
            "J_arm_r_02": -0.5,   # 右肩  
            "J_hip_l_pitch": -0.2, # 左髋俯仰
            "J_hip_r_pitch": -0.2, # 右髋俯仰
            "J_knee_l_pitch": 0.4,  # 左膝
            "J_knee_r_pitch": 0.4   # 右膝
        }
        
        for joint_name, target_pos in position_targets.items():
            self.control_loop.set_position_hold_target(joint_name, target_pos)
        
        # 启动控制循环
        self.control_loop.start()
        
        # 运行指定时间
        start_time = time.time()
        while time.time() - start_time < duration:
            # 推进仿真
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.001)
        
        # 停止控制循环
        self.control_loop.stop()
        
        # 打印统计信息
        self.control_loop.print_status()
        self._print_sensor_data()
    
    def run_balance_test(self, duration: float = 15.0):
        """运行平衡控制测试"""
        print(f"\n=== Running BALANCE test for {duration} seconds ===")
        
        # 设置平衡控制模式
        self.control_loop.set_control_mode(ControlLoopMode.BALANCE)
        
        # 调整平衡控制增益
        self.control_loop.set_balance_gains(
            kp_pos=150.0,
            kd_pos=15.0,
            kp_ori=80.0,
            kd_ori=8.0
        )
        
        # 启动控制循环
        self.control_loop.start()
        
        # 运行指定时间，在中途施加扰动
        start_time = time.time()
        disturbance_applied = False
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            # 在5秒时施加扰动力
            if 4.5 < current_time < 6.0 and not disturbance_applied:
                # 施加侧向扰动力
                self.data.xfrc_applied[1, 0] = 50.0  # X方向50N推力
                disturbance_applied = True
                print("Applied lateral disturbance force")
            elif current_time > 6.0:
                # 移除扰动力
                self.data.xfrc_applied[1, 0] = 0.0
            
            # 推进仿真
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.001)
        
        # 停止控制循环
        self.control_loop.stop()
        
        # 打印统计信息
        self.control_loop.print_status()
        self._print_sensor_data()
    
    def run_with_visualization(self, mode: ControlLoopMode = ControlLoopMode.BALANCE, 
                              duration: float = 30.0):
        """运行带可视化的控制循环"""
        print(f"\n=== Running {mode.value} with visualization for {duration} seconds ===")
        
        # 设置控制模式
        self.control_loop.set_control_mode(mode)
        
        if mode == ControlLoopMode.POSITION_HOLD:
            # 设置位置目标
            position_targets = {
                "J_arm_l_02": 0.3,
                "J_arm_r_02": -0.3,
                "J_hip_l_pitch": -0.1,
                "J_hip_r_pitch": -0.1
            }
            for joint_name, target_pos in position_targets.items():
                self.control_loop.set_position_hold_target(joint_name, target_pos)
        
        # 启动控制循环
        self.control_loop.start()
        
        # 启动可视化
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                last_status_time = start_time
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    # 推进仿真
                    mujoco.mj_step(self.model, self.data)
                    
                    # 更新可视化
                    viewer.sync()
                    
                    # 每5秒打印一次状态
                    if time.time() - last_status_time > 5.0:
                        print(f"Running... Time: {time.time() - start_time:.1f}s")
                        self._print_brief_status()
                        last_status_time = time.time()
                    
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("Visualization interrupted by user")
        
        # 停止控制循环
        self.control_loop.stop()
        
        # 最终状态报告
        self.control_loop.print_status()
    
    def run_frequency_test(self, target_frequencies: list = [100, 500, 1000, 2000]):
        """测试不同控制频率的性能"""
        print("\n=== Running frequency performance test ===")
        
        for freq in target_frequencies:
            print(f"\nTesting {freq} Hz control frequency...")
            
            # 创建新的控制循环
            test_loop = ControlLoop(
                mujoco_model=self.model,
                mujoco_data=self.data,
                control_frequency=freq,
                enable_monitoring=True
            )
            
            # 设置空闲模式
            test_loop.set_control_mode(ControlLoopMode.IDLE)
            
            # 运行3秒测试
            test_loop.start()
            test_duration = 3.0
            start_time = time.time()
            
            while time.time() - start_time < test_duration:
                mujoco.mj_step(self.model, self.data)
                time.sleep(0.0001)  # 0.1ms仿真步长
            
            test_loop.stop()
            
            # 打印性能统计
            stats = test_loop.get_status()
            print(f"  Target: {freq} Hz, Actual: {stats.get('loop_frequency', 0):.1f} Hz")
            print(f"  Avg loop time: {stats.get('avg_loop_time', 0)*1000:.2f} ms")
            print(f"  Max loop time: {stats.get('max_loop_time', 0)*1000:.2f} ms")
    
    def _print_sensor_data(self):
        """打印传感器数据摘要"""
        print("\n--- Sensor Data Summary ---")
        
        # IMU数据
        imu_ori = self.data_bus.get_imu_orientation()
        imu_vel = self.data_bus.get_imu_angular_velocity()
        print(f"IMU Orientation (w,x,y,z): ({imu_ori.w:.3f}, {imu_ori.x:.3f}, {imu_ori.y:.3f}, {imu_ori.z:.3f})")
        print(f"IMU Angular Vel (x,y,z): ({imu_vel.x:.3f}, {imu_vel.y:.3f}, {imu_vel.z:.3f})")
        
        # 质心数据
        com_pos = self.data_bus.get_center_of_mass_position()
        com_vel = self.data_bus.get_center_of_mass_velocity()
        print(f"CoM Position (x,y,z): ({com_pos.x:.3f}, {com_pos.y:.3f}, {com_pos.z:.3f})")
        print(f"CoM Velocity (x,y,z): ({com_vel.x:.3f}, {com_vel.y:.3f}, {com_vel.z:.3f})")
        
        # 足部接触状态
        left_foot_contact = self.data_bus.get_end_effector_contact_state("left_foot")
        right_foot_contact = self.data_bus.get_end_effector_contact_state("right_foot")
        print(f"Foot Contact - Left: {left_foot_contact.name if left_foot_contact else 'None'}, Right: {right_foot_contact.name if right_foot_contact else 'None'}")
    
    def _print_brief_status(self):
        """打印简要状态"""
        stats = self.control_loop.get_status()
        print(f"  Mode: {stats['current_mode']}, Freq: {stats.get('loop_frequency', 0):.1f} Hz, Loops: {stats['loop_count']}")


def main():
    """主函数 - 演示控制循环的各种功能"""
    print("=== MuJoCo Control Loop Example ===")
    
    try:
        # 创建示例实例
        example = MuJoCoControlExample()
        
        # 选择要运行的测试
        print("\nSelect test to run:")
        print("1. IDLE mode test (5 seconds)")
        print("2. Position Hold test (10 seconds)")
        print("3. Balance Control test (15 seconds)")
        print("4. Balance Control with Visualization (30 seconds)")
        print("5. Frequency Performance test")
        print("6. Run all tests sequentially")
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == "1":
            example.run_idle_test()
        elif choice == "2":
            example.run_position_hold_test()
        elif choice == "3":
            example.run_balance_test()
        elif choice == "4":
            example.run_with_visualization(ControlLoopMode.BALANCE)
        elif choice == "5":
            example.run_frequency_test()
        elif choice == "6":
            # 运行所有测试
            example.run_idle_test(3.0)
            time.sleep(1.0)
            example.run_position_hold_test(5.0)
            time.sleep(1.0)
            example.run_balance_test(8.0)
            time.sleep(1.0)
            example.run_frequency_test([100, 500, 1000])
        else:
            print("Invalid choice, running balance test with visualization")
            example.run_with_visualization(ControlLoopMode.BALANCE)
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    main() 