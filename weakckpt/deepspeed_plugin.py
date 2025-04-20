import deepspeed
import torch
from .manager import WeakCkptManager
from .config import WeakCkptConfig

class WeakCkptDeepSpeedPlugin:
    """
    DeepSpeed 集成插件：在训练启动前创建 WeakCkptManager，
    并绑定 DeepSpeed 引擎的 save_checkpoint/load_checkpoint 钩子
    """
    def __init__(self, config: WeakCkptConfig, save_dir: str):
        self.manager = WeakCkptManager(save_dir, config)

    def attach_to_engine(self, engine: deepspeed.DeepSpeedEngine):
        # 在每个训练 step 结束后调用
        engine.register_post_step_callback(self._post_step)
        # 重写 DeepSpeed 的 save_checkpoint
        self._orig_save = engine.save_checkpoint
        engine.save_checkpoint = self._ds_save_checkpoint.__get__(engine, type(engine))

    def _post_step(self, engine):
        # 获取模型 state dict 并 snapshot
        state = engine.module.state_dict()
        self.manager.on_step_end(state)

    def _ds_save_checkpoint(self, engine, checkpoint_dir, client_state=None):
        # 先调用原生 save
        ret = self._orig_save(checkpoint_dir, client_state)
        # 触发 weakckpt 持久化
        self.manager.ckpt.persist(self.manager.next_ckpt_id)
        return ret