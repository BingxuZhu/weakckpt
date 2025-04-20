import os
import torch

class WeakCkptCheckpoint:
    """
    负责跨步快照（snapshot）和持久化（persist）逻辑
    """
    def __init__(self, manager, config):
        self.manager = manager
        self.config = config
        self.parts = []

    def snapshot(self, state: dict, step: int) -> None:
        """截取当前训练步的部分参数"""
        part = self._select_part(state, step)
        self.parts.append(part)

    def persist(self, ckpt_id: int) -> None:
        """当累积分片数量>=stride_steps时，合并并写盘"""
        if len(self.parts) >= self.config.stride_steps:
            merged = {}
            for p in self.parts:
                merged.update(p)
            path = os.path.join(
                self.manager.save_dir, f"weakckpt_{ckpt_id}.pt"
            )
            torch.save(merged, path)
            print(f"[WeakCkpt] Saved ckpt {ckpt_id} at step {self.manager.current_step}")
            self.parts.clear()

    def _select_part(self, state: dict, step: int) -> dict:
        """按键排序分割 state_dict，轮询取第 idx 片"""
        keys = sorted(state.keys())
        total = len(keys)
        m = total // self.config.stride_steps
        idx = (step - 1) % self.config.stride_steps
        selected_keys = keys[idx * m: (idx + 1) * m]
        return {k: state[k] for k in selected_keys}