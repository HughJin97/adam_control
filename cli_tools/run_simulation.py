import argparse
import mujoco as mj
import mujoco.viewer
from ..sim.gait_sim import GaitSimulation

def main():
    parser = argparse.ArgumentParser(description='运行步态仿真')
    parser.add_argument('--model', type=str, default='models/scene.xml',
                      help='MJCF模型文件路径')
    args = parser.parse_args()
    
    # 初始化仿真环境
    sim = GaitSimulation(args.model)
    
    # 启动viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # 设置viewer
        sim.set_viewer(viewer)
        
        # 主循环
        while viewer.is_running():
            # 步进仿真
            sim.step()
            mj.mj_step(sim.model, sim.data)
            
            # 同步viewer
            viewer.sync()
            
if __name__ == '__main__':
    main() 