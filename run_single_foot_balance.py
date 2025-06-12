#!/usr/bin/env python3
"""
单脚平衡仿真启动脚本
快速启动Mujoco单脚平衡仿真
"""

import sys
import os

# 添加sim目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.single_foot_balance_simulation import SingleFootBalanceSimulation

def main():
    print("=== 单脚平衡仿真 ===")
    print("在Mujoco环境中让机器人保持单脚支撑静止，调用MPC计算平衡")
    print()
    
    # 选择支撑脚
    while True:
        choice = input("选择支撑脚 (l=左脚, r=右脚, 默认=右脚): ").lower().strip()
        if choice in ['l', 'left', '左']:
            support_foot = 'left'
            break
        elif choice in ['r', 'right', '右', '']:
            support_foot = 'right'
            break
        else:
            print("请输入 'l' 或 'r'")
    
    # 仿真时长
    while True:
        try:
            duration_input = input("仿真时长 (秒, 默认=10): ").strip()
            if duration_input == '':
                duration = 10.0
            else:
                duration = float(duration_input)
            if duration > 0:
                break
            else:
                print("时长必须大于0")
        except ValueError:
            print("请输入有效的数字")
    
    # 是否显示可视化
    while True:
        viewer_choice = input("显示3D可视化界面? (y/n, 默认=y): ").lower().strip()
        if viewer_choice in ['y', 'yes', '是', '']:
            use_viewer = True
            break
        elif viewer_choice in ['n', 'no', '否']:
            use_viewer = False
            break
        else:
            print("请输入 'y' 或 'n'")
    
    # 是否绘制结果
    while True:
        plot_choice = input("仿真结束后绘制结果图表? (y/n, 默认=y): ").lower().strip()
        if plot_choice in ['y', 'yes', '是', '']:
            show_plot = True
            break
        elif plot_choice in ['n', 'no', '否']:
            show_plot = False
            break
        else:
            print("请输入 'y' 或 'n'")
    
    print(f"\n开始仿真...")
    print(f"支撑脚: {'左脚' if support_foot == 'left' else '右脚'}")
    print(f"仿真时长: {duration}秒")
    print(f"可视化: {'开启' if use_viewer else '关闭'}")
    print(f"结果图表: {'开启' if show_plot else '关闭'}")
    print()
    
    # 创建并运行仿真
    try:
        sim = SingleFootBalanceSimulation()
        sim.support_foot = support_foot
        sim.target_joint_positions = sim._get_single_foot_stance_pose()
        
        # 运行仿真
        sim.run_simulation(duration=duration, use_viewer=use_viewer)
        
        # 绘制结果
        if show_plot:
            sim.plot_results()
            
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    except Exception as e:
        print(f"\n仿真出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 