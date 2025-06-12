import numpy as np
import mujoco as mj

class RobotController:
    def __init__(self, model, data):
        """初始化机器人控制器
        
        Args:
            model: MjModel实例
            data: MjData实例
        """
        self.model = model
        self.data = data
        
        # PD控制器参数
        self.kp = 100.0  # 位置增益
        self.kd = 10.0   # 速度增益
        
    def compute_pd_torques(self, q_target, qd_target=None):
        """计算PD控制的输出力矩
        
        Args:
            q_target: 目标关节角度
            qd_target: 目标关节速度，默认为0
            
        Returns:
            tau: 计算得到的控制力矩
        """
        if qd_target is None:
            qd_target = np.zeros_like(q_target)
            
        q_error = q_target - self.data.qpos
        qd_error = qd_target - self.data.qvel
        
        return self.kp * q_error + self.kd * qd_error 