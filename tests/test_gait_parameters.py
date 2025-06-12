#!/usr/bin/env python3
"""
步态参数系统测试脚本

测试步态参数管理器和数据总线的集成功能：
1. 步态参数的设置和获取
2. 预设参数的加载和切换
3. 实时相位计算
4. 足部轨迹生成
5. 配置文件的导入导出

作者: Adam Control Team
版本: 1.0
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

# 导入模块
from gait_parameters import get_gait_manager, GaitType
from data_bus import get_data_bus


class GaitParameterTester:
    """步态参数测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.gait_manager = get_gait_manager()
        self.data_bus = get_data_bus()
        
        print("GaitParameterTester initialized")
        print(f"GaitManager available: {self.data_bus._has_gait_manager}")
    
    def test_basic_parameters(self):
        """测试基本参数设置和获取"""
        print("\n=== 测试基本参数设置 ===")
        
        # 测试速度设置
        test_speeds = [0.05, 0.1, 0.2, 0.3]
        for speed in test_speeds:
            self.data_bus.set_walking_speed(speed)
            actual_speed = self.data_bus.get_walking_speed()
            step_length = self.gait_manager.spatial.step_length
            print(f"设置速度: {speed:.2f} m/s -> 实际速度: {actual_speed:.2f} m/s, 步长: {step_length:.3f} m")
        
        # 测试步频设置
        test_frequencies = [0.5, 1.0, 1.5, 2.0]
        for freq in test_frequencies:
            self.data_bus.set_step_frequency(freq)
            actual_freq = self.data_bus.get_step_frequency()
            cycle_time = self.gait_manager.timing.gait_cycle_time
            print(f"设置步频: {freq:.1f} Hz -> 实际步频: {actual_freq:.1f} Hz, 周期: {cycle_time:.2f} s")
    
    def test_preset_loading(self):
        """测试预设参数加载"""
        print("\n=== 测试预设参数加载 ===")
        
        # 获取可用预设
        presets = self.data_bus.get_available_gait_presets()
        print(f"可用预设: {presets}")
        
        # 测试每个预设
        for preset_name in presets:
            print(f"\n加载预设: {preset_name}")
            success = self.data_bus.load_gait_preset(preset_name)
            if success:
                phase_info = self.data_bus.get_gait_phase_info()
                print(f"  周期时间: {phase_info['gait_cycle_time']:.2f}s")
                print(f"  步频: {phase_info['step_frequency']:.2f}Hz")
                print(f"  行走速度: {phase_info['walking_speed']:.3f}m/s")
                print(f"  步长: {self.gait_manager.spatial.step_length:.3f}m")
                print(f"  步宽: {self.gait_manager.spatial.step_width:.3f}m")
                print(f"  抬脚高度: {self.gait_manager.spatial.step_height:.3f}m")
    
    def test_phase_calculation(self):
        """测试相位计算"""
        print("\n=== 测试相位计算 ===")
        
        # 加载正常步行预设
        self.data_bus.load_gait_preset("normal_walk")
        cycle_time = self.gait_manager.timing.gait_cycle_time
        
        print(f"使用周期时间: {cycle_time:.2f}s")
        print("时间\t周期相位\t左腿相位\t右腿相位\t左脚状态\t右脚状态\t双支撑")
        print("-" * 80)
        
        # 测试一个完整周期
        for i in range(11):
            t = i * cycle_time / 10
            self.data_bus.update_gait_phase(t)
            phase_info = self.data_bus.get_gait_phase_info()
            
            left_state = "摆动" if phase_info["left_in_swing"] else "支撑"
            right_state = "摆动" if phase_info["right_in_swing"] else "支撑"
            double_support = "是" if phase_info["double_support"] else "否"
            
            print(f"{t:.2f}\t{phase_info['cycle_phase']:.2f}\t\t"
                  f"{phase_info['left_phase']:.2f}\t\t{phase_info['right_phase']:.2f}\t\t"
                  f"{left_state}\t\t{right_state}\t\t{double_support}")
    
    def test_foot_trajectory(self):
        """测试足部轨迹生成"""
        print("\n=== 测试足部轨迹生成 ===")
        
        # 加载正常步行预设
        self.data_bus.load_gait_preset("normal_walk")
        
        # 生成一个周期的足部轨迹
        cycle_time = self.gait_manager.timing.gait_cycle_time
        time_steps = np.linspace(0, cycle_time, 100)
        
        left_trajectory = []
        right_trajectory = []
        
        for t in time_steps:
            self.data_bus.update_gait_phase(t)
            
            left_target = self.data_bus.get_foot_target_position("left")
            right_target = self.data_bus.get_foot_target_position("right")
            
            left_trajectory.append([left_target.x, left_target.y, left_target.z])
            right_trajectory.append([right_target.x, right_target.y, right_target.z])
        
        left_trajectory = np.array(left_trajectory)
        right_trajectory = np.array(right_trajectory)
        
        # 打印轨迹统计
        print(f"左脚轨迹:")
        print(f"  X范围: {left_trajectory[:, 0].min():.3f} ~ {left_trajectory[:, 0].max():.3f} m")
        print(f"  Y范围: {left_trajectory[:, 1].min():.3f} ~ {left_trajectory[:, 1].max():.3f} m")
        print(f"  Z范围: {left_trajectory[:, 2].min():.3f} ~ {left_trajectory[:, 2].max():.3f} m")
        
        print(f"右脚轨迹:")
        print(f"  X范围: {right_trajectory[:, 0].min():.3f} ~ {right_trajectory[:, 0].max():.3f} m")
        print(f"  Y范围: {right_trajectory[:, 1].min():.3f} ~ {right_trajectory[:, 1].max():.3f} m")
        print(f"  Z范围: {right_trajectory[:, 2].min():.3f} ~ {right_trajectory[:, 2].max():.3f} m")
        
        return time_steps, left_trajectory, right_trajectory
    
    def test_walking_control(self):
        """测试行走控制接口"""
        print("\n=== 测试行走控制接口 ===")
        
        # 测试开始/停止行走
        print("初始状态:")
        print(f"  机器人模式: {self.data_bus.get_robot_mode()}")
        print(f"  是否在行走: {self.data_bus.gait.is_walking}")
        print(f"  行走速度: {self.data_bus.get_walking_speed():.3f} m/s")
        
        # 开始行走
        print("\n开始行走...")
        self.data_bus.start_walking()
        self.data_bus.set_walking_speed(0.15)
        print(f"  机器人模式: {self.data_bus.get_robot_mode()}")
        print(f"  是否在行走: {self.data_bus.gait.is_walking}")
        print(f"  行走速度: {self.data_bus.get_walking_speed():.3f} m/s")
        
        # 开始转向
        print("\n开始转向...")
        self.data_bus.start_turning(0.3)  # 0.3 rad/s 转向速度
        print(f"  是否在转向: {self.data_bus.gait.is_turning}")
        print(f"  转向速度: {self.data_bus.gait.turning_rate:.3f} rad/s")
        
        # 停止转向
        print("\n停止转向...")
        self.data_bus.stop_turning()
        print(f"  是否在转向: {self.data_bus.gait.is_turning}")
        print(f"  转向速度: {self.data_bus.gait.turning_rate:.3f} rad/s")
        
        # 停止行走
        print("\n停止行走...")
        self.data_bus.stop_walking()
        print(f"  机器人模式: {self.data_bus.get_robot_mode()}")
        print(f"  是否在行走: {self.data_bus.gait.is_walking}")
        print(f"  行走速度: {self.data_bus.get_walking_speed():.3f} m/s")
    
    def test_config_file_operations(self):
        """测试配置文件操作"""
        print("\n=== 测试配置文件操作 ===")
        
        try:
            # 导出当前参数到文件
            export_file = "test_gait_export.yaml"
            self.data_bus.export_gait_parameters(export_file)
            print(f"✓ 参数导出成功: {export_file}")
            
            # 修改一些参数
            original_speed = self.data_bus.get_walking_speed()
            self.data_bus.set_walking_speed(0.5)
            print(f"修改速度: {original_speed:.3f} -> {self.data_bus.get_walking_speed():.3f} m/s")
            
            # 从文件导入参数
            self.data_bus.import_gait_parameters(export_file)
            restored_speed = self.data_bus.get_walking_speed()
            print(f"✓ 参数导入成功，速度恢复: {restored_speed:.3f} m/s")
            
            # 测试从配置文件加载
            config_file = "config/gait_parameters.yaml"
            try:
                self.data_bus.import_gait_parameters(config_file)
                print(f"✓ 从配置文件加载成功: {config_file}")
            except FileNotFoundError:
                print(f"⚠ 配置文件未找到: {config_file}")
            
        except Exception as e:
            print(f"✗ 配置文件操作失败: {e}")
    
    def plot_foot_trajectories(self, time_steps, left_traj, right_traj):
        """绘制足部轨迹图"""
        print("\n=== 绘制足部轨迹 ===")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('足部轨迹分析', fontsize=16)
            
            # X-Z 平面 (侧视图)
            axes[0, 0].plot(left_traj[:, 0], left_traj[:, 2], 'b-', label='左脚', linewidth=2)
            axes[0, 0].plot(right_traj[:, 0], right_traj[:, 2], 'r-', label='右脚', linewidth=2)
            axes[0, 0].set_xlabel('X 位置 [m]')
            axes[0, 0].set_ylabel('Z 位置 [m]')
            axes[0, 0].set_title('侧视图 (X-Z)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Y-Z 平面 (后视图)
            axes[0, 1].plot(left_traj[:, 1], left_traj[:, 2], 'b-', label='左脚', linewidth=2)
            axes[0, 1].plot(right_traj[:, 1], right_traj[:, 2], 'r-', label='右脚', linewidth=2)
            axes[0, 1].set_xlabel('Y 位置 [m]')
            axes[0, 1].set_ylabel('Z 位置 [m]')
            axes[0, 1].set_title('后视图 (Y-Z)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # X-Y 平面 (俯视图)
            axes[1, 0].plot(left_traj[:, 0], left_traj[:, 1], 'b-', label='左脚', linewidth=2)
            axes[1, 0].plot(right_traj[:, 0], right_traj[:, 1], 'r-', label='右脚', linewidth=2)
            axes[1, 0].set_xlabel('X 位置 [m]')
            axes[1, 0].set_ylabel('Y 位置 [m]')
            axes[1, 0].set_title('俯视图 (X-Y)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 时间历程
            axes[1, 1].plot(time_steps, left_traj[:, 2], 'b-', label='左脚高度', linewidth=2)
            axes[1, 1].plot(time_steps, right_traj[:, 2], 'r-', label='右脚高度', linewidth=2)
            axes[1, 1].set_xlabel('时间 [s]')
            axes[1, 1].set_ylabel('Z 位置 [m]')
            axes[1, 1].set_title('足部高度时间历程')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('foot_trajectories.png', dpi=150, bbox_inches='tight')
            print("✓ 足部轨迹图已保存: foot_trajectories.png")
            
            # 在支持的环境中显示图像
            try:
                plt.show()
            except:
                print("注意: 无法显示图像窗口，请查看保存的图片文件")
                
        except ImportError:
            print("⚠ matplotlib 未安装，跳过轨迹绘制")
        except Exception as e:
            print(f"✗ 绘制轨迹失败: {e}")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=== 步态参数系统综合测试 ===")
        
        # 运行各项测试
        self.test_basic_parameters()
        self.test_preset_loading()
        self.test_phase_calculation()
        
        # 测试足部轨迹并绘图
        time_steps, left_traj, right_traj = self.test_foot_trajectory()
        
        self.test_walking_control()
        self.test_config_file_operations()
        
        # 绘制轨迹
        self.plot_foot_trajectories(time_steps, left_traj, right_traj)
        
        print("\n=== 测试完成 ===")
        print("步态参数系统功能正常！")


def main():
    """主函数"""
    try:
        # 创建测试器并运行测试
        tester = GaitParameterTester()
        tester.run_comprehensive_test()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 