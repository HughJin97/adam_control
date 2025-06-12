#!/usr/bin/env python3
"""
步态可视化启动器

提供多种可视化选项：
1. 3D MuJoCo仿真（推荐）
2. 2D matplotlib监控
3. 测试模式

作者: Adam Control Team
"""

import sys
import subprocess
import argparse
import platform
from typing import Optional

def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    步态可视化系统                              ║
║                  AzureLoong Robot Gait                       ║
╠══════════════════════════════════════════════════════════════╣
║  实时显示:                                                   ║
║  • 当前legState（颜色/文字标注）                              ║
║  • target_foot_pos（小球/透明标记）                          ║
║  • 足底力变化                                                ║
║  • 步态状态转换                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """检查依赖项"""
    print("检查依赖项...")
    
    deps_status = {}
    
    # 检查MuJoCo
    try:
        import mujoco
        deps_status['mujoco'] = f"✓ MuJoCo {mujoco.__version__}"
    except ImportError:
        deps_status['mujoco'] = "✗ MuJoCo 未安装"
    
    # 检查matplotlib
    try:
        import matplotlib
        deps_status['matplotlib'] = f"✓ matplotlib {matplotlib.__version__}"
    except ImportError:
        deps_status['matplotlib'] = "✗ matplotlib 未安装"
    
    # 检查numpy
    try:
        import numpy
        deps_status['numpy'] = f"✓ numpy {numpy.__version__}"
    except ImportError:
        deps_status['numpy'] = "✗ numpy 未安装"
    
    # 检查步态模块
    try:
        from gait_scheduler import GaitScheduler
        from foot_placement import FootPlacementPlanner
        from data_bus import DataBus
        deps_status['gait_modules'] = "✓ 步态模块已加载"
    except ImportError as e:
        deps_status['gait_modules'] = f"✗ 步态模块错误: {e}"
    
    # 打印依赖状态
    print("\n依赖项状态:")
    print("─" * 50)
    for dep, status in deps_status.items():
        print(f"  {status}")
    print("─" * 50)
    
    return deps_status

def show_menu():
    """显示菜单选项"""
    menu = """
请选择可视化模式:

🎮 [1] 3D MuJoCo仿真 (推荐)
   • 完整3D物理仿真环境
   • 实时机器人模型显示
   • 交互式控制 (键盘/鼠标)
   • 状态颜色实时更新
   • 目标位置3D标记

📊 [2] 2D matplotlib监控
   • 实时数据图表显示
   • 步态状态时间序列
   • 足底力变化曲线
   • 足部轨迹平面图
   • 状态信息面板

🔧 [3] 测试模式
   • 命令行输出
   • 性能测试
   • 调试用途

📸 [4] 离屏渲染模式
   • 无GUI运行
   • 适用于服务器
   • 性能监控

❓ [5] 显示帮助信息

🚪 [0] 退出

"""
    print(menu)

def get_python_command():
    """获取合适的Python命令"""
    # 在macOS上，如果有mjpython，优先使用
    if platform.system() == "Darwin":
        # 检查是否有mjpython
        try:
            result = subprocess.run(["which", "mjpython"], capture_output=True, text=True)
            if result.returncode == 0:
                return "mjpython"
        except:
            pass
    
    return sys.executable

def run_mujoco_simulation(args: Optional[list] = None):
    """运行MuJoCo 3D仿真"""
    print("启动3D MuJoCo仿真...")
    
    python_cmd = get_python_command()
    cmd = [python_cmd, "gait_visualization_simulation.py"]
    if args:
        cmd.extend(args)
    
    # 在macOS上提示使用mjpython
    if platform.system() == "Darwin" and python_cmd != "mjpython":
        print("\n⚠️  注意: 在macOS上运行MuJoCo GUI推荐使用mjpython")
        print("如果遇到问题，请尝试:")
        print("  1. 安装mjpython: pip install mujoco")
        print("  2. 使用命令: mjpython gait_visualization_simulation.py")
        print("  3. 或使用离屏模式: python gait_visualization_simulation.py --offscreen\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"仿真启动失败: {e}")
        if "launch_passive" in str(e) and platform.system() == "Darwin":
            print("\n💡 提示: 请使用mjpython运行，或选择离屏渲染模式")
        return False
    except FileNotFoundError:
        print("错误: 找不到 gait_visualization_simulation.py")
        return False

def run_matplotlib_monitor(args: Optional[list] = None):
    """运行matplotlib 2D监控"""
    print("启动2D matplotlib监控...")
    
    cmd = [sys.executable, "gait_state_monitor.py"]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"监控启动失败: {e}")
        return False
    except FileNotFoundError:
        print("错误: 找不到 gait_state_monitor.py")
        return False

def run_test_mode():
    """运行测试模式"""
    print("启动测试模式...")
    
    cmd = [sys.executable, "gait_visualization_simulation.py", "--test"]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        return False

def run_offscreen_mode():
    """运行离屏渲染模式"""
    print("启动离屏渲染模式...")
    
    cmd = [sys.executable, "gait_visualization_simulation.py", "--offscreen"]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"离屏渲染失败: {e}")
        return False

def show_help():
    """显示帮助信息"""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║                         使用说明                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 🎮 3D MuJoCo仿真控制:                                        ║
║   • 空格键: 暂停/继续                                        ║
║   • R: 重置仿真                                              ║  
║   • S: 开始步行                                              ║
║   • +/-: 调节速度                                            ║
║   • ESC: 退出                                                ║
║   • 鼠标: 旋转/缩放视角                                       ║
║                                                              ║
║ 📊 2D matplotlib监控:                                        ║
║   • 左上: 步态状态时间序列                                    ║
║   • 右上: 足底力变化曲线                                      ║
║   • 左下: 足部轨迹平面图                                      ║
║   • 右下: 实时状态信息                                        ║
║   • Ctrl+C: 退出监控                                         ║
║                                                              ║
║ 🔧 命令行选项:                                               ║
║   --model <path>     指定模型文件                             ║
║   --offscreen        离屏渲染模式                             ║
║   --test             测试模式                                 ║
║   --history <num>    历史数据长度                             ║
║   --interval <ms>    更新间隔                                 ║
║                                                              ║
║ 💡 macOS用户注意:                                            ║
║   GUI模式需要使用mjpython运行:                                ║
║   mjpython gait_visualization_simulation.py                  ║
║                                                              ║
║ 📁 相关文件:                                                 ║
║   • gait_visualization_simulation.py  3D仿真脚本             ║
║   • gait_state_monitor.py            2D监控脚本              ║
║   • README_GaitVisualization.md      详细文档                ║
║                                                              ║
║ 🎯 功能测试标准:                                             ║
║   • 连续20个步态周期无跳变/卡死                               ║
║   • 步态周期误差 ≤ ±5%                                      ║
║   • 落脚点偏差 < 2cm                                         ║
║   • 提前触地可正确切换状态                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(help_text)

def install_dependencies():
    """安装缺失的依赖项"""
    print("安装缺失的依赖项...")
    
    packages = ["mujoco", "matplotlib", "numpy"]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True, check=True)
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 安装失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="步态可视化启动器")
    parser.add_argument("--auto-install", action="store_true", 
                       help="自动安装缺失的依赖项")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], 
                       help="直接指定模式 (1:3D, 2:2D, 3:测试, 4:离屏)")
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 检查依赖项
    deps = check_dependencies()
    
    # 自动安装依赖项
    if args.auto_install:
        if "✗" in str(deps):
            install_dependencies()
            deps = check_dependencies()
    
    # 检查关键依赖
    if "✗" in deps.get('gait_modules', ''):
        print("\n❌ 错误: 步态模块未找到或有错误")
        print("请确保以下文件存在:")
        print("  - gait_scheduler.py")
        print("  - foot_placement.py") 
        print("  - data_bus.py")
        return 1
    
    # 直接模式
    if args.mode:
        print(f"\n直接启动模式 {args.mode}...")
        
        if args.mode == 1:
            if "✗" in deps.get('mujoco', ''):
                print("❌ MuJoCo未安装，无法使用3D仿真模式")
                return 1
            return 0 if run_mujoco_simulation() else 1
        
        elif args.mode == 2:
            if "✗" in deps.get('matplotlib', ''):
                print("❌ matplotlib未安装，无法使用2D监控模式")
                return 1
            return 0 if run_matplotlib_monitor() else 1
        
        elif args.mode == 3:
            return 0 if run_test_mode() else 1
            
        elif args.mode == 4:
            return 0 if run_offscreen_mode() else 1
    
    # 交互模式
    while True:
        show_menu()
        
        try:
            choice = input("请输入选择 [0-5]: ").strip()
            
            if choice == "0":
                print("感谢使用步态可视化系统！")
                break
            
            elif choice == "1":
                if "✗" in deps.get('mujoco', ''):
                    print("❌ MuJoCo未安装，请先安装: pip install mujoco")
                    continue
                
                print("正在启动3D MuJoCo仿真...")
                run_mujoco_simulation()
            
            elif choice == "2":
                if "✗" in deps.get('matplotlib', ''):
                    print("❌ matplotlib未安装，请先安装: pip install matplotlib")
                    continue
                
                print("正在启动2D matplotlib监控...")
                run_matplotlib_monitor()
            
            elif choice == "3":
                run_test_mode()
            
            elif choice == "4":
                run_offscreen_mode()
                
            elif choice == "5":
                show_help()
            
            else:
                print("❌ 无效选择，请输入0-5之间的数字")
        
        except KeyboardInterrupt:
            print("\n\n感谢使用步态可视化系统！")
            break
        except EOFError:
            print("\n\n感谢使用步态可视化系统！")
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 