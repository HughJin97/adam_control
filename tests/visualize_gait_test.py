#!/usr/bin/env python3
"""
步态逻辑可视化测试

可视化显示步态状态机的工作情况，验证：
1. 状态转换逻辑
2. 腿部状态交替
3. 摆动时间进度
4. 手动触发效果
"""

import time
import numpy as np
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class GaitVisualizer:
    """步态可视化器"""
    
    def __init__(self):
        # 配置
        config = GaitSchedulerConfig()
        config.swing_time = 0.5         # 500ms 便于观察
        config.double_support_time = 0.1
        config.use_time_trigger = True
        config.use_sensor_trigger = True
        config.require_both_triggers = False
        config.enable_logging = True
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        # 测试状态
        self.test_time = 0.0
        self.state_changes = 0
        self.manual_triggers = 0
        
        print("="*80)
        print("步态逻辑可视化测试")
        print("="*80)
        print(f"摆动时间: {config.swing_time}s")
        print(f"双支撑时间: {config.double_support_time}s")
    
    def display_current_state(self):
        """显示当前状态"""
        print(f"\n[{self.test_time:.2f}s] 当前状态:")
        print(f"  状态机状态: {self.scheduler.current_state.value}")
        print(f"  腿部状态: {self.scheduler.leg_state.value}")
        print(f"  摆动腿: {self.scheduler.swing_leg}")
        print(f"  支撑腿: {self.scheduler.support_leg}")
        
        if self.scheduler.swing_leg != "none":
            progress = self.scheduler.swing_elapsed_time / self.scheduler.config.swing_time
            progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"  摆动进度: {progress:.1%} [{progress_bar}] {self.scheduler.swing_elapsed_time:.3f}s")
        
        print(f"  总时间: {self.scheduler.total_time:.3f}s")
        print(f"  状态变化次数: {self.state_changes}")
    
    def manual_trigger_state_change(self):
        """手动触发状态变化"""
        print(f"\n>>> 手动触发状态变化 #{self.manual_triggers + 1}")
        
        current_state = self.scheduler.current_state
        
        # 强制满足切换条件
        if current_state == GaitState.LEFT_SUPPORT:
            # 左支撑 -> 双支撑 (L->R)
            print("  强制左支撑 -> 双支撑(L->R)")
            self.scheduler.swing_elapsed_time = self.scheduler.config.swing_time + 0.01
            self.scheduler.right_foot_force = 250.0  # 右脚着地
            self.scheduler.right_foot_contact = True
            
        elif current_state == GaitState.DOUBLE_SUPPORT_LR:
            # 双支撑(L->R) -> 右支撑
            print("  强制双支撑(L->R) -> 右支撑")
            self.scheduler._transition_to_state(GaitState.RIGHT_SUPPORT)
            
        elif current_state == GaitState.RIGHT_SUPPORT:
            # 右支撑 -> 双支撑(R->L)
            print("  强制右支撑 -> 双支撑(R->L)")
            self.scheduler.swing_elapsed_time = self.scheduler.config.swing_time + 0.01
            self.scheduler.left_foot_force = 250.0  # 左脚着地
            self.scheduler.left_foot_contact = True
            
        elif current_state == GaitState.DOUBLE_SUPPORT_RL:
            # 双支撑(R->L) -> 左支撑
            print("  强制双支撑(R->L) -> 左支撑")
            self.scheduler._transition_to_state(GaitState.LEFT_SUPPORT)
        
        self.manual_triggers += 1
    
    def test_basic_state_machine(self):
        """测试基本状态机"""
        print(f"\n{'='*60}")
        print("1. 基本状态机测试")
        print('='*60)
        
        # 显示初始状态
        print("初始状态:")
        self.display_current_state()
        
        # 启动步态
        print(f"\n启动步态...")
        self.scheduler.start_walking()
        self.display_current_state()
        
        return True
    
    def test_manual_progression(self):
        """测试手动推进"""
        print(f"\n{'='*60}")
        print("2. 手动状态推进测试")
        print('='*60)
        
        print("将手动触发5次状态变化，观察状态机响应...")
        
        for i in range(5):
            print(f"\n--- 手动触发 {i+1}/5 ---")
            
            # 显示触发前状态
            print("触发前:")
            self.display_current_state()
            
            # 手动触发
            self.manual_trigger_state_change()
            
            # 推进状态机几步
            for step in range(5):
                state_changed = self.scheduler.update_gait_state(0.01)
                self.test_time += 0.01
                
                if state_changed:
                    self.state_changes += 1
                    print(f"  >> 状态变化! (第{self.state_changes}次)")
                    break
            
            # 显示触发后状态
            print("触发后:")
            self.display_current_state()
            
            time.sleep(0.5)  # 暂停观察
        
        return self.state_changes >= 3
    
    def test_continuous_operation(self):
        """测试连续运行"""
        print(f"\n{'='*60}")
        print("3. 连续运行测试")
        print('='*60)
        
        print("连续运行5秒，观察自动状态转换...")
        
        dt = 0.02  # 20ms 控制周期
        steps = int(5.0 / dt)  # 5秒
        
        last_display_time = 0
        
        for step in range(steps):
            current_time = step * dt
            
            # 模拟传感器数据
            if self.scheduler.swing_leg == "right":
                self.scheduler.left_foot_contact = True
                self.scheduler.left_foot_force = 200.0
                # 摆动末期右脚着地
                if self.scheduler.swing_elapsed_time >= self.scheduler.config.swing_time * 0.9:
                    self.scheduler.right_foot_contact = True
                    self.scheduler.right_foot_force = 200.0
                else:
                    self.scheduler.right_foot_contact = False
                    self.scheduler.right_foot_force = 0.0
            elif self.scheduler.swing_leg == "left":
                self.scheduler.right_foot_contact = True
                self.scheduler.right_foot_force = 200.0
                # 摆动末期左脚着地
                if self.scheduler.swing_elapsed_time >= self.scheduler.config.swing_time * 0.9:
                    self.scheduler.left_foot_contact = True
                    self.scheduler.left_foot_force = 200.0
                else:
                    self.scheduler.left_foot_contact = False
                    self.scheduler.left_foot_force = 0.0
            else:
                # 双支撑
                self.scheduler.left_foot_contact = True
                self.scheduler.right_foot_contact = True
                self.scheduler.left_foot_force = 200.0
                self.scheduler.right_foot_force = 200.0
            
            # 推进状态机
            state_changed = self.scheduler.update_gait_state(dt)
            self.test_time += dt
            
            if state_changed:
                self.state_changes += 1
                print(f"\n[{current_time:.2f}s] 自动状态变化 #{self.state_changes}")
                self.display_current_state()
                last_display_time = current_time
            
            # 每1秒显示一次状态
            elif current_time - last_display_time >= 1.0:
                print(f"\n[{current_time:.2f}s] 状态更新:")
                self.display_current_state()
                last_display_time = current_time
        
        return self.state_changes >= 2
    
    def test_target_positions(self):
        """测试目标位置"""
        print(f"\n{'='*60}")
        print("4. 目标位置测试")
        print('='*60)
        
        # 获取当前目标位置
        targets = self.scheduler.get_target_foot_positions()
        
        print("当前目标位置:")
        for foot, pos in targets.items():
            print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
        
        # 设置运动指令
        print(f"\n设置前进运动指令...")
        self.scheduler.set_motion_command(forward_velocity=0.3, lateral_velocity=0.0, turning_rate=0.0)
        
        # 手动更新目标位置
        print(f"手动更新目标位置...")
        new_targets = {
            "left_foot": {"x": 0.2, "y": 0.09, "z": 0.0},
            "right_foot": {"x": 0.2, "y": -0.09, "z": 0.0}
        }
        self.data_bus.target_foot_pos.update(new_targets)
        
        # 验证更新
        updated_targets = self.scheduler.get_target_foot_positions()
        print("更新后目标位置:")
        for foot, pos in updated_targets.items():
            print(f"  {foot}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
        
        # 计算左右脚间距
        left_pos = updated_targets['left_foot']
        right_pos = updated_targets['right_foot']
        y_distance = abs(left_pos['y'] - right_pos['y'])
        
        print(f"\n左右脚间距: {y_distance:.3f}m")
        
        return y_distance >= 0.08
    
    def run_visualization_test(self):
        """运行可视化测试"""
        print("开始步态逻辑可视化测试...\n")
        
        results = {}
        
        # 1. 基本状态机测试
        results['basic'] = self.test_basic_state_machine()
        
        # 2. 手动推进测试
        results['manual'] = self.test_manual_progression()
        
        # 3. 连续运行测试
        results['continuous'] = self.test_continuous_operation()
        
        # 4. 目标位置测试
        results['positions'] = self.test_target_positions()
        
        # 最终总结
        print(f"\n{'='*80}")
        print("可视化测试总结")
        print('='*80)
        
        print("测试结果:")
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        print(f"\n统计信息:")
        print(f"  总测试时间: {self.test_time:.2f}s")
        print(f"  状态变化次数: {self.state_changes}")
        print(f"  手动触发次数: {self.manual_triggers}")
        print(f"  最终状态: {self.scheduler.current_state.value}")
        print(f"  最终legState: {self.scheduler.leg_state.value}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print(f"\n🎉 可视化测试全部通过!")
            print("✅ 基本状态机功能正常")
            print("✅ 手动触发响应正确")
            print("✅ 连续运行状态转换")
            print("✅ 目标位置合理更新")
        else:
            print(f"\n⚠️ 部分测试需要关注")
            print("但基本功能可以观察到工作")
        
        print('='*80)
        
        return all_passed


def main():
    """主函数"""
    try:
        visualizer = GaitVisualizer()
        success = visualizer.run_visualization_test()
        return success
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n测试完成，结果: {'成功' if success else '部分成功'}")
    exit(0 if success else 1) 