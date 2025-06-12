#!/usr/bin/env python3
"""
单脚平衡仿真演示脚本
展示在Mujoco环境中机器人单脚支撑平衡的效果
"""

import sys
import os
import time

# 添加sim目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim'))

from sim.simple_single_foot_balance import SimpleSingleFootBalanceSimulation

def demo_single_foot_balance():
    """演示单脚平衡仿真"""
    
    print("=" * 60)
    print("🤖 单脚平衡仿真演示")
    print("=" * 60)
    print()
    print("本演示将展示机器人在Mujoco环境中的单脚支撑平衡能力")
    print("使用简化的平衡控制算法来维持机器人的稳定性")
    print()
    
    # 演示配置
    demos = [
        {
            'name': '右脚支撑平衡',
            'support_foot': 'right',
            'duration': 5.0,
            'description': '机器人使用右脚支撑，左脚抬起，保持平衡'
        },
        {
            'name': '左脚支撑平衡', 
            'support_foot': 'left',
            'duration': 5.0,
            'description': '机器人使用左脚支撑，右脚抬起，保持平衡'
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"📍 演示 {i}: {demo['name']}")
        print(f"   {demo['description']}")
        print(f"   仿真时长: {demo['duration']}秒")
        print()
        
        # 询问是否运行此演示
        while True:
            choice = input(f"是否运行演示 {i}? (y/n/s=跳过): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                run_demo = True
                break
            elif choice in ['n', 'no', '否']:
                print("演示结束")
                return
            elif choice in ['s', 'skip', '跳过']:
                run_demo = False
                break
            else:
                print("请输入 'y' (运行), 'n' (结束), 或 's' (跳过)")
        
        if not run_demo:
            print(f"跳过演示 {i}")
            print()
            continue
        
        # 询问是否显示可视化
        while True:
            viewer_choice = input("显示3D可视化界面? (y/n, 推荐=y): ").lower().strip()
            if viewer_choice in ['y', 'yes', '是', '']:
                use_viewer = True
                break
            elif viewer_choice in ['n', 'no', '否']:
                use_viewer = False
                break
            else:
                print("请输入 'y' 或 'n'")
        
        print(f"\n🚀 开始运行演示 {i}...")
        print("-" * 40)
        
        try:
            # 创建仿真实例
            sim = SimpleSingleFootBalanceSimulation()
            sim.support_foot = demo['support_foot']
            sim.target_joint_positions = sim._get_single_foot_stance_pose()
            
            # 运行仿真
            sim.run_simulation(
                duration=demo['duration'],
                use_viewer=use_viewer
            )
            
            print(f"✅ 演示 {i} 完成")
            
            # 询问是否绘制结果
            while True:
                plot_choice = input("查看结果图表? (y/n): ").lower().strip()
                if plot_choice in ['y', 'yes', '是']:
                    sim.plot_results()
                    break
                elif plot_choice in ['n', 'no', '否']:
                    break
                else:
                    print("请输入 'y' 或 'n'")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  演示 {i} 被用户中断")
        except Exception as e:
            print(f"\n❌ 演示 {i} 出错: {e}")
        
        print()
        
        # 如果不是最后一个演示，询问是否继续
        if i < len(demos):
            while True:
                continue_choice = input("继续下一个演示? (y/n): ").lower().strip()
                if continue_choice in ['y', 'yes', '是']:
                    break
                elif continue_choice in ['n', 'no', '否']:
                    print("演示结束")
                    return
                else:
                    print("请输入 'y' 或 'n'")
            print()
    
    print("🎉 所有演示完成！")
    print()
    print("📊 演示总结:")
    print("- 展示了机器人的单脚支撑平衡能力")
    print("- 使用简化的平衡控制算法")
    print("- 在Mujoco物理仿真环境中运行")
    print("- 实时控制频率: 125Hz")
    print("- 支持左脚和右脚支撑模式")
    print()
    print("💡 技术特点:")
    print("- PD控制器维持关节目标角度")
    print("- 踝关节平衡控制补偿质心偏移")
    print("- 实时状态监控和数据记录")
    print("- 可视化结果分析")

def quick_demo():
    """快速演示模式"""
    print("🚀 快速演示模式")
    print("将运行一个5秒的右脚支撑平衡演示")
    print()
    
    try:
        sim = SimpleSingleFootBalanceSimulation()
        sim.support_foot = 'right'
        sim.target_joint_positions = sim._get_single_foot_stance_pose()
        
        # 运行仿真
        sim.run_simulation(duration=5.0, use_viewer=True)
        
        print("✅ 快速演示完成")
        
        # 自动显示结果
        sim.plot_results()
        
    except KeyboardInterrupt:
        print("\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")

def main():
    """主函数"""
    print("🤖 单脚平衡仿真演示系统")
    print()
    print("选择演示模式:")
    print("1. 完整演示 (包含左脚和右脚支撑)")
    print("2. 快速演示 (仅右脚支撑，5秒)")
    print("3. 退出")
    print()
    
    while True:
        choice = input("请选择 (1/2/3): ").strip()
        if choice == '1':
            demo_single_foot_balance()
            break
        elif choice == '2':
            quick_demo()
            break
        elif choice == '3':
            print("再见！")
            break
        else:
            print("请输入 1, 2 或 3")

if __name__ == "__main__":
    main() 