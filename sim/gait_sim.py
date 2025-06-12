from ..gait_core.data_bus import DataBus
from ..gait_core.scheduler import GaitScheduler
from .visualization import GaitVisualizer
from .model_loader import ModelLoader
from .controller import RobotController

class GaitSimulation:
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化步态仿真环境
        
        Args:
            model_path: MJCF模型文件路径
        """
        # 加载模型
        self.model_loader = ModelLoader(model_path)
        self.model = self.model_loader.model
        self.data = self.model_loader.data
        
        # 初始化控制器
        self.controller = RobotController(self.model, self.data)
        
        # 初始化步态组件
        self.data_bus = DataBus()
        self.scheduler = GaitScheduler(self.data_bus)
        
        # 可视化组件在viewer设置后初始化
        self.visualizer = None
        
    def set_viewer(self, viewer):
        """设置MuJoCo viewer
        
        Args:
            viewer: MuJoCo viewer实例
        """
        self.visualizer = GaitVisualizer(viewer)
        
    def step(self):
        """执行一个仿真步骤"""
        # 更新步态规划
        self.scheduler.update()
        
        # 获取目标状态
        targets = self.scheduler.get_targets()
        
        # 计算控制输出
        tau = self.controller.compute_pd_torques(targets.q, targets.qd)
        self.data.ctrl[:] = tau
        
        # 更新可视化
        if self.visualizer:
            self.visualizer.visualize_foot_targets(
                targets.left_foot_target,
                targets.right_foot_target,
                self.data_bus.get_left_foot_pos(),
                self.data_bus.get_right_foot_pos()
            ) 