import mujoco as mj

class ModelLoader:
    def __init__(self, model_path: str = "models/scene.xml"):
        """初始化MuJoCo模型加载器
        
        Args:
            model_path: MJCF模型文件路径
        """
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
    
    def reset(self):
        """重置模型状态"""
        mj.mj_resetData(self.model, self.data) 