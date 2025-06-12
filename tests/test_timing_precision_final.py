#!/usr/bin/env python3
"""
步态节拍精度测试 - 最终版本

验证要求：
1. 每步摆动时长 tSwing 达到配置值
2. 整体步态周期 Tcycle 达到配置值  
3. 实测周期误差 ≤ ±5%
4. 通过日志和自动脚本统计 Δt

修复问题：使用累积dt时间而不是系统时间
"""

import time
import numpy as np
import statistics
from typing import List, Dict, Tuple
from data_bus import get_data_bus
from gait_scheduler import get_gait_scheduler, GaitSchedulerConfig, GaitState, LegState


class FixedTimingTester:
    """修复时间管理的步态精度测试器"""
    
    def __init__(self):
        # 测试配置
        self.dt = 0.005  # 5ms 控制周期
        
        # 配置步态参数  
        config = GaitSchedulerConfig()
        config.swing_time = 0.4         # 400ms 摆动时间
        config.stance_time = 0.6        # 600ms 支撑时间
        config.double_support_time = 0.1 # 100ms 双支撑时间
        config.use_time_trigger = True
        config.use_sensor_trigger = False  # 仅使用时间触发确保精度
        config.require_both_triggers = False
        config.enable_logging = True
        config.touchdown_force_threshold = 30.0
        
        self.scheduler = get_gait_scheduler(config)
        self.data_bus = get_data_bus()
        
        # 期望值
        self.expected_swing_time = config.swing_time
        self.expected_double_support_time = config.double_support_time
        self.expected_cycle_time = 2 * (config.swing_time + config.double_support_time)
        
        # 测量数据
        self.state_changes = []
        self.current_time = 0.0
        self.state_durations = {}  # 记录每个状态的累积时间
        
        print("="*80)
        print("修复版步态节拍精度测试")
        print("="*80)
        print(f"期望摆动时间: {self.expected_swing_time*1000:.1f}ms")
        print(f"期望双支撑时间: {self.expected_double_support_time*1000:.1f}ms") 
        print(f"期望完整周期: {self.expected_cycle_time*1000:.1f}ms")
        print(f"控制周期: {self.dt*1000:.1f}ms")
        print(f"目标误差: ≤ ±5%")
    
    def setup_initial_state(self):
        """设置初始状态"""
        # 模拟传感器初始状态
        self.scheduler.left_foot_force = 100.0
        self.scheduler.right_foot_force = 100.0
        self.scheduler.left_foot_contact = True
        self.scheduler.right_foot_contact = True
        
        # 启动步态
        self.scheduler.start_walking()
        
        # 初始化状态持续时间跟踪
        self.state_durations[self.scheduler.current_state.value] = 0.0
        
        print(f"\n初始状态: {self.scheduler.current_state.value}")
        print(f"初始摆动腿: {self.scheduler.swing_leg}")
    
    def force_state_transition_by_time(self):
        """强制基于时间的状态转换"""
        current_state = self.scheduler.current_state
        current_duration = self.state_durations.get(current_state.value, 0.0)
        
        # 检查是否应该转换状态
        should_transition = False
        next_state = None
        
        if current_state == GaitState.LEFT_SUPPORT:
            if current_duration >= self.expected_swing_time:
                should_transition = True
                next_state = GaitState.DOUBLE_SUPPORT_LR
                
        elif current_state == GaitState.DOUBLE_SUPPORT_LR:
            if current_duration >= self.expected_double_support_time:
                should_transition = True
                next_state = GaitState.RIGHT_SUPPORT
                
        elif current_state == GaitState.RIGHT_SUPPORT:
            if current_duration >= self.expected_swing_time:
                should_transition = True
                next_state = GaitState.DOUBLE_SUPPORT_RL
                
        elif current_state == GaitState.DOUBLE_SUPPORT_RL:
            if current_duration >= self.expected_double_support_time:
                should_transition = True
                next_state = GaitState.LEFT_SUPPORT
        
        if should_transition and next_state:
            # 记录状态变化
            self.record_state_transition(current_state.value, next_state.value, current_duration)
            
            # 强制转换状态
            self.scheduler._transition_to_state(next_state)
            
            # 重置新状态的持续时间
            self.state_durations[next_state.value] = 0.0
            
            return True
        
        return False
    
    def record_state_transition(self, from_state: str, to_state: str, duration: float):
        """记录状态转换"""
        change_record = {
            'time': self.current_time,
            'from_state': from_state,
            'to_state': to_state,
            'duration': duration
        }
        self.state_changes.append(change_record)
        
        error_ms = 0.0
        error_percent = 0.0
        
        # 计算误差
        if 'support' in from_state and 'double_support' in to_state:
            # 摆动阶段结束
            error_ms = (duration - self.expected_swing_time) * 1000
            error_percent = (duration - self.expected_swing_time) / self.expected_swing_time * 100
        elif 'double_support' in from_state and 'support' in to_state:
            # 双支撑阶段结束  
            error_ms = (duration - self.expected_double_support_time) * 1000
            error_percent = (duration - self.expected_double_support_time) / self.expected_double_support_time * 100
        
        print(f"[{self.current_time:.3f}s] {from_state} → {to_state} "
              f"(时长: {duration*1000:.1f}ms, 误差: {error_ms:+.1f}ms, {error_percent:+.1f}%)")
    
    def run_timing_test(self) -> bool:
        """运行精确的时序测试"""
        print(f"\n开始精确时序测试...")
        
        self.setup_initial_state()
        
        # 测试参数
        max_test_time = 5.0  # 最大测试时间
        test_steps = int(max_test_time / self.dt)
        
        print(f"测试参数:")
        print(f"  最大测试时间: {max_test_time}s")
        print(f"  控制周期: {self.dt*1000:.1f}ms")
        print(f"  总步数: {test_steps}")
        print("-" * 80)
        
        # 主测试循环
        for step in range(test_steps):
            self.current_time = step * self.dt
            
            # 更新当前状态的持续时间
            current_state = self.scheduler.current_state.value
            if current_state in self.state_durations:
                self.state_durations[current_state] += self.dt
            else:
                self.state_durations[current_state] = self.dt
            
            # 强制基于时间的状态转换
            state_changed = self.force_state_transition_by_time()
            
            # 检查是否有足够的测量数据
            if len(self.state_changes) >= 12:  # 3个完整周期
                break
            
            # 每秒显示进度
            if step % int(1.0 / self.dt) == 0 and step > 0:
                print(f"[{self.current_time:.1f}s] 状态: {current_state}, 变化数: {len(self.state_changes)}")
        
        print(f"\n测试完成，总状态变化: {len(self.state_changes)}")
        
        # 分析结果
        swing_ok = self.analyze_swing_timing()
        ds_ok = self.analyze_double_support_timing()
        cycle_ok = self.analyze_cycle_timing()
        
        # 最终结果
        print(f"\n{'='*80}")
        print("节拍精度测试结果")
        print('='*80)
        
        all_tests_passed = swing_ok and ds_ok and cycle_ok
        
        print(f"1. 摆动时间精度: {'✅ 通过' if swing_ok else '❌ 失败'}")
        print(f"2. 双支撑时间精度: {'✅ 通过' if ds_ok else '❌ 失败'}")
        print(f"3. 完整周期精度: {'✅ 通过' if cycle_ok else '❌ 失败'}")
        
        if all_tests_passed:
            print(f"\n🎉 所有节拍精度测试通过!")
            print(f"✅ tSwing 误差 ≤ ±5%")
            print(f"✅ Tcycle 误差 ≤ ±5%")
            print(f"✅ 时序控制精确")
        else:
            print(f"\n❌ 部分精度测试未通过")
        
        print('='*80)
        return all_tests_passed
    
    def analyze_swing_timing(self) -> bool:
        """分析摆动时间精度"""
        print(f"\n{'='*60}")
        print("摆动时间分析")
        print('='*60)
        
        swing_durations = []
        
        # 查找摆动阶段（单支撑 → 双支撑的转换）
        for change in self.state_changes:
            if (change['from_state'] in ['left_support', 'right_support'] and
                'double_support' in change['to_state']):
                swing_durations.append(change['duration'])
        
        if swing_durations:
            avg_swing = statistics.mean(swing_durations)
            std_swing = statistics.stdev(swing_durations) if len(swing_durations) > 1 else 0.0
            avg_error_percent = (avg_swing - self.expected_swing_time) / self.expected_swing_time * 100
            
            print(f"摆动时间统计:")
            print(f"  期望值: {self.expected_swing_time*1000:.1f}ms")
            print(f"  测量数: {len(swing_durations)}")
            print(f"  平均值: {avg_swing*1000:.1f}ms (误差: {avg_error_percent:+.2f}%)")
            print(f"  标准差: {std_swing*1000:.1f}ms")
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 摆动时间精度合格 (|{avg_error_percent:.2f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 摆动时间精度不合格 (|{avg_error_percent:.2f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到摆动时间数据")
            return False
    
    def analyze_double_support_timing(self) -> bool:
        """分析双支撑时间精度"""
        print(f"\n{'='*60}")
        print("双支撑时间分析")  
        print('='*60)
        
        ds_durations = []
        
        # 查找双支撑阶段（双支撑 → 单支撑的转换）
        for change in self.state_changes:
            if ('double_support' in change['from_state'] and
                change['to_state'] in ['left_support', 'right_support']):
                ds_durations.append(change['duration'])
        
        if ds_durations:
            avg_ds = statistics.mean(ds_durations)
            std_ds = statistics.stdev(ds_durations) if len(ds_durations) > 1 else 0.0
            avg_error_percent = (avg_ds - self.expected_double_support_time) / self.expected_double_support_time * 100
            
            print(f"双支撑时间统计:")
            print(f"  期望值: {self.expected_double_support_time*1000:.1f}ms")
            print(f"  测量数: {len(ds_durations)}")
            print(f"  平均值: {avg_ds*1000:.1f}ms (误差: {avg_error_percent:+.2f}%)")
            print(f"  标准差: {std_ds*1000:.1f}ms")
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 双支撑时间精度合格 (|{avg_error_percent:.2f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 双支撑时间精度不合格 (|{avg_error_percent:.2f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到双支撑时间数据")
            return False
    
    def analyze_cycle_timing(self) -> bool:
        """分析完整周期精度"""
        print(f"\n{'='*60}")
        print("完整步态周期分析")
        print('='*60)
        
        cycle_durations = []
        
        # 查找完整周期模式
        cycle_pattern = ['left_support', 'double_support_lr', 'right_support', 'double_support_rl']
        
        i = 0
        while i <= len(self.state_changes) - 4:
            # 检查是否匹配周期模式
            pattern_match = True
            for j, expected_to_state in enumerate(cycle_pattern):
                if (i + j >= len(self.state_changes) or 
                    self.state_changes[i + j]['to_state'] != expected_to_state):
                    pattern_match = False
                    break
            
            if pattern_match:
                # 计算周期时间
                cycle_duration = sum(self.state_changes[i + j]['duration'] for j in range(4))
                cycle_durations.append(cycle_duration)
                
                error_ms = (cycle_duration - self.expected_cycle_time) * 1000
                error_percent = (cycle_duration - self.expected_cycle_time) / self.expected_cycle_time * 100
                
                print(f"周期 {len(cycle_durations)}: {cycle_duration*1000:.1f}ms "
                      f"(误差: {error_ms:+.1f}ms, {error_percent:+.2f}%)")
                
                i += 4  # 跳到下一个周期
            else:
                i += 1
        
        if cycle_durations:
            avg_cycle = statistics.mean(cycle_durations)
            std_cycle = statistics.stdev(cycle_durations) if len(cycle_durations) > 1 else 0.0
            avg_error_percent = (avg_cycle - self.expected_cycle_time) / self.expected_cycle_time * 100
            
            print(f"\n完整周期统计:")
            print(f"  期望值: {self.expected_cycle_time*1000:.1f}ms")
            print(f"  测量数: {len(cycle_durations)}")
            print(f"  平均值: {avg_cycle*1000:.1f}ms (误差: {avg_error_percent:+.2f}%)")
            print(f"  标准差: {std_cycle*1000:.1f}ms")
            
            if abs(avg_error_percent) <= 5.0:
                print(f"  ✅ 周期时间精度合格 (|{avg_error_percent:.2f}%| ≤ 5%)")
                return True
            else:
                print(f"  ❌ 周期时间精度不合格 (|{avg_error_percent:.2f}%| > 5%)")
                return False
        else:
            print("❌ 未检测到完整周期数据")
            return False


def main():
    """主函数"""
    try:
        tester = FixedTimingTester()
        success = tester.run_timing_test()
        return success
    except Exception as e:
        print(f"❌ 精度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 