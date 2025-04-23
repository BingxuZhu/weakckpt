import os
import time
import threading
import logging
import torch
from typing import Dict, List, Optional

from .config import WeakCkptConfig
from .checkpoint import WeakCkptCheckpoint
from .disk_bw import get_storage_bandwidth

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DynamicScheduler:
    """
    Dynamic scheduler: adjusts checkpoint trigger interval based on I/O bandwidth,
    error metrics, and training load.
    """
    def __init__(self, config: WeakCkptConfig):
        self.config = config
        self.last_adjust_time = time.time()
        self.current_interval = config.base_interval

    def update(self, bandwidth: float, error_metric: float) -> int:
        """
        Update the checkpoint interval:
          - shorten interval if bandwidth is below threshold
          - shorten interval if error exceeds max_version_diff to reduce drift
        """
        now = time.time()
        # Avoid updating too frequently
        if now - self.last_adjust_time < 10:
            return self.current_interval

        interval = self.config.base_interval
        # If I/O bandwidth low, checkpoint more often
        if bandwidth < self.config.io_bw_threshold:
            interval = max(1, interval // 2)
        # If version error too high, checkpoint more often
        if error_metric > self.config.max_version_diff:
            interval = max(1, interval // 2)

        self.current_interval = interval
        self.last_adjust_time = now
        logger.debug(f"Scheduler updated: interval={interval}, bw={bandwidth:.2f}, err={error_metric:.2f}")
        return interval


class CascadedRecovery:
    """
    Cascaded recovery: build a multi-version index to quickly restore parameters
    from historical checkpoints.
    """
    def __init__(self, manager_dir: str):
        self.manager_dir = manager_dir
        self.index = {}  # mapping checkpoint_id -> set of parameter keys
        self._build_index()

    def _build_index(self):
        # Scan checkpoint files and record contained parameter keys
        files = sorted(f for f in os.listdir(self.manager_dir) if f.startswith('weakckpt_'))
        for fname in files:
            ckpt_id = int(fname.split('_')[1].split('.')[0])
            path = os.path.join(self.manager_dir, fname)
            try:
                state = torch.load(path, map_location='cpu')
                self.index[ckpt_id] = set(state.keys())
                logger.debug(f"Indexed checkpoint {ckpt_id}: {len(state)} parameters")
            except Exception as e:
                logger.warning(f"Failed to index {fname}: {e}")

    def recover(self, target_id: Optional[int] = None) -> Dict:
        """
        Perform cascaded recovery:
          - if target_id not specified, use highest available id
          - load parameters from newest to oldest to reconstruct full state
        """
        if not self.index:
            raise RuntimeError("No checkpoints available for recovery")

        if target_id is None:
            target_id = max(self.index.keys())

        reconstructed = {}
        # iterate from latest to earliest
        for ckpt_id in sorted(self.index.keys(), reverse=True):
            if ckpt_id > target_id:
                continue
            path = os.path.join(self.manager_dir, f"weakckpt_{ckpt_id}.pt")
            state = torch.load(path, map_location='cpu')
            # fill missing parameters
            for key, val in state.items():
                if key not in reconstructed:
                    reconstructed[key] = val
            # stop if we have recovered all keys for target checkpoint
            if reconstructed.keys() >= self.index[target_id]:
                break

        logger.info(f"Recovered {len(reconstructed)} parameters up to checkpoint {target_id}")
        return reconstructed


class WeakCkptManager:
    """
    Weak consistency checkpoint manager:
      - handles staggered snapshot and persist
      - dynamic trigger logic
      - cascaded recovery
    """

    def __init__(self, save_dir: str, config: WeakCkptConfig):
        self.save_dir = save_dir
        self.config = config
        os.makedirs(self.save_dir, exist_ok=True)

        self.ckpt = WeakCkptCheckpoint(self, config)
        self.scheduler = DynamicScheduler(config)
        self.current_step = 0
        self.next_ckpt_id = 0
        self._deepspeed_engine = None

        self._error_metric = 0.0
        self._lock = threading.Lock()
        logger.info(f"WeakCkptManager initialized: dir={save_dir}, config={config.__dict__}")

    def on_step_end(self, state_dict: Dict):
        """
        Call at the end of each training step:
          - compute version error metric
          - perform staggered snapshot and attempt persist
          - update trigger interval dynamically
        """
        with self._lock:
            self.current_step += 1
            # estimate error based on pending parts
            self._error_metric = self._compute_error_metric()

            # perform snapshot of this step's partition
            self.ckpt.snapshot(state_dict, self.current_step)

            # measure I/O bandwidth and update interval
            bandwidth = get_storage_bandwidth(self.save_dir)
            interval = self.scheduler.update(bandwidth, self._error_metric)

            # attempt to persist if enough parts accumulated
            self.ckpt.persist(self.next_ckpt_id)

            # trigger new checkpoint id if interval reached
            if self.current_step % interval == 0:
                logger.info(f"Triggering checkpoint {self.next_ckpt_id} at step {self.current_step}")
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

    def _compute_error_metric(self) -> float:
        """
        Simple version drift metric: ratio of pending parts to stride_steps
        """
        pending = len(self.ckpt.parts)
        metric = pending / max(1, self.config.stride_steps)
        logger.debug(f"Error metric: {metric:.2f} (pending parts={pending})")
        return metric

    def recover(self, checkpoint_id: Optional[int] = None) -> Dict:
        """
        Recover model state using cascaded recovery mechanism
        """
        logger.info(f"Starting cascaded recovery up to checkpoint {checkpoint_id}")
        cascader = CascadedRecovery(self.save_dir)
        return cascader.recover(target_id=checkpoint_id)

    def save_meta(self):
        """
        Save manager metadata: current step and next checkpoint id
        """
        meta = {'current_step': self.current_step, 'next_ckpt_id': self.next_ckpt_id}
        path = os.path.join(self.save_dir, 'weakckpt_meta.pt')
        torch.save(meta, path)
        logger.debug(f"Saved manager metadata to {path}")

    def load_meta(self):
        """
        Load manager metadata and restore internal state
        """
        path = os.path.join(self.save_dir, 'weakckpt_meta.pt')
        if os.path.exists(path):
            meta = torch.load(path, map_location='cpu')
            self.current_step = meta.get('current_step', 0)
            self.next_ckpt_id = meta.get('next_ckpt_id', 0)
            logger.info(f"Loaded metadata: {meta}")
        else:
            logger.warning("No manager metadata found; starting fresh")

    def __repr__(self):
        return (f"<WeakCkptManager step={self.current_step} next_id={self.next_ckpt_id} "
                f"interval={self.scheduler.current_interval}>")