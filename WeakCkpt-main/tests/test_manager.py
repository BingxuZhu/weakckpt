import os
import torch
import tempfile
from weakckpt.config import WeakCkptConfig
from weakckpt.manager import WeakCkptManager


def test_checkpoint_cycle():
    cfg = WeakCkptConfig(stride_steps=2, base_interval=4, io_bw_threshold=1e9)
    with tempfile.TemporaryDirectory() as d:
        mgr = WeakCkptManager(d, cfg)
        # 模拟10步训练，每步state只有{'a': tensor(step)}
        for i in range(1, 11):
            mgr.on_step_end({'a': torch.tensor(i)})
        files = sorted(os.listdir(d))
        assert any(f.startswith('weakckpt_') for f in files)


def test_recover():
    cfg = WeakCkptConfig(stride_steps=1, base_interval=1, io_bw_threshold=0)
    with tempfile.TemporaryDirectory() as d:
        mgr = WeakCkptManager(d, cfg)
        for i in range(3):
            mgr.on_step_end({'x': torch.tensor(i)})
        state = mgr.recover()
        assert 'x' in state