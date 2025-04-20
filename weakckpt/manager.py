import os
import torch
from .checkpoint import WeakCkptCheckpoint
from .disk_bw import get_storage_bandwidth
from .utils import compute_error_threshold

class WeakCkptManager:
    """
    管理弱一致性检查点全流程：跨步保存、动态触发、级联恢复
    """
    def __init__(self, save_dir: str, config):
        self.save_dir = save_dir
        self.config = config
        os.makedirs(save_dir, exist_ok=True)
        self.ckpt = WeakCkptCheckpoint(self, config)
        self.current_step = 0
        self.next_ckpt_id = 0
        self._deepspeed_engine = None

    def on_step_end(self, state_dict: dict):
        """在每个训练步结束时调用"""
        self.current_step += 1
        # 分片快照
        self.ckpt.snapshot(state_dict, self.current_step)
        # 持久化（跨步完成时写盘）
        self.ckpt.persist(self.next_ckpt_id)
        # 判断是否触发下一个完整检查点ID
        if self._should_trigger():
            self.next_ckpt_id += 1
            
    def bind_deepspeed(self, engine):
        """
        绑定 DeepSpeed engine，以支持从 DeepSpeed checkpoint 恢复
        """
        self._deepspeed_engine = engine
        # 在恢复时调用 manager.recover 并加载 state
        # 例如，在用户代码中：
        # engine.load_checkpoint(...)
        # 然后 manager.recover()

    def _should_trigger(self) -> bool:
        """根据动态频率决定何时增加 checkpoint ID"""
        bw = get_storage_bandwidth(self.save_dir)
        interval = compute_error_threshold(self.config, bw)
        return self.current_step % interval == 0

    def recover(self, checkpoint_dir: str = None) -> dict:
        """
        从历史检查点中恢复最新模型状态。
        TODO: 实现级联索引恢复（多版本融合、误差校正等）
        返回合并后的 state_dict。
        """
        checkpoint_dir = checkpoint_dir or self.save_dir
        files = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith('weakckpt_') and f.endswith('.pt')
        )
        state = {}
        for fname in files:
            part = torch.load(os.path.join(checkpoint_dir, fname))
            state.update(part)
        return state