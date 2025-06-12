import numpy as np
import mujoco as mj

class GaitVisualizer:
    def __init__(self, viewer):
        """初始化步态可视化器
        
        Args:
            viewer: MuJoCo viewer实例
        """
        self.viewer = viewer
        
    def visualize_foot_targets(self, left_target, right_target, left_pos, right_pos):
        """可视化目标落脚点和当前足端位置
        
        Args:
            left_target: 左脚目标位置 [x, y, z]
            right_target: 右脚目标位置 [x, y, z]
            left_pos: 左脚当前位置 [x, y, z]
            right_pos: 右脚当前位置 [x, y, z]
        """
        with self.viewer.lock():
            # 清空自定义场景
            self.viewer.user_scn.ngeom = 0
            geom_idx = 0
            
            def _add_sphere(pos, rgba):
                nonlocal geom_idx
                if geom_idx >= len(self.viewer.user_scn.geoms):
                    return
                    
                g = self.viewer.user_scn.geoms[geom_idx]
                g.type = mj.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [0.05, 0, 0]  # 球体半径
                g.pos[:] = pos
                g.rgba[:] = rgba
                geom_idx += 1
                
            # 目标点(半透明)
            _add_sphere(left_target, [1, 0, 0, 0.5])   # 红色
            _add_sphere(right_target, [0, 1, 0, 0.5])  # 绿色
            
            # 实际位置(不透明)
            _add_sphere(left_pos, [0, 0, 1, 1])   # 蓝色
            _add_sphere(right_pos, [0, 0, 1, 1])  # 蓝色 