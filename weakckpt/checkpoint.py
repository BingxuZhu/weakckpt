import os
import threading
import torch
import logging
from typing import Dict, List, Optional, Callable, Any
from queue import Queue, Empty

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PartitionStrategy:
    """
    Strategy to split a model's state_dict into staggered partitions.
    """
    def __init__(self, stride_steps: int):
        self.stride_steps = stride_steps

    def partition(self, state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Split the state_dict into `stride_steps` partitions based on key order.
        """
        keys = sorted(state.keys())
        total = len(keys)
        base = total // self.stride_steps
        parts: List[Dict[str, torch.Tensor]] = []
        for i in range(self.stride_steps):
            start = i * base
            end = (i + 1) * base if i < self.stride_steps - 1 else total
            part_state = {k: state[k] for k in keys[start:end]}
            parts.append(part_state)
        return parts

class AsyncWriter(threading.Thread):
    """
    Background thread to persist checkpoint partitions asynchronously.
    """
    def __init__(self, save_dir: str, batch_size: int = 1, compress: bool = False):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.queue: Queue = Queue()
        self.running = True
        self.batch_size = batch_size
        self.compress = compress
        self._setup()

    def _setup(self):
        os.makedirs(self.save_dir, exist_ok=True)
        logger.debug(f"AsyncWriter setup at {self.save_dir}, compress={self.compress}")

    def run(self):
        logger.info("AsyncWriter thread started")
        while self.running or not self.queue.empty():
            try:
                ckpt_id, part = self.queue.get(timeout=1)
            except Empty:
                continue
            try:
                filename = f"weakckpt_{ckpt_id}.pt"
                path = os.path.join(self.save_dir, filename)
                if self.compress:
                    # placeholder for compression logic
                    torch.save(part, path)
                else:
                    torch.save(part, path)
                logger.info(f"[AsyncWriter] saved checkpoint {ckpt_id} ({len(part)} params)")
            except Exception as error:
                logger.error(f"[AsyncWriter] error saving {ckpt_id}: {error}")
            finally:
                self.queue.task_done()

    def enqueue(self, ckpt_id: int, part: Dict[str, torch.Tensor]):
        """Enqueue a partition for writing."""
        try:
            safe_part = {k: v.clone().cpu() for k, v in part.items()}
            self.queue.put((ckpt_id, safe_part))
            logger.debug(f"[AsyncWriter] enqueued part for ckpt {ckpt_id}")
        except Exception as e:
            logger.error(f"AsyncWriter enqueue failed: {e}")

    def stop(self):
        """Stop the writer after flushing queue."""
        logger.info("Stopping AsyncWriter thread...")
        self.running = False
        self.queue.join()
        self.join()
        logger.info("AsyncWriter stopped")

class WeakCkptCheckpoint:
    """
    Handles staggered snapshot and asynchronous persistence of model parameters.
    Includes support for custom hooks, metrics collection, and error handling.
    """
    def __init__(self,
                 manager,
                 config,
                 pre_hook: Optional[Callable[[int, List[str]], None]] = None,
                 post_hook: Optional[Callable[[int], None]] = None):
        self.manager = manager
        self.config = config
        self.strategy = PartitionStrategy(config.stride_steps)
        self.current_parts: List[Dict[str, torch.Tensor]] = []
        self.writer = AsyncWriter(manager.save_dir,
                                  batch_size=config.stride_steps,
                                  compress=getattr(config, 'compress', False))
        self.writer.start()
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.metrics: Dict[str, Any] = {}
        logger.info("WeakCkptCheckpoint initialized")

    def snapshot(self, state: Dict[str, torch.Tensor], step: int) -> None:
        """
        On the first step of each stride cycle, compute partitions.
        Subsequent steps only reuse those partitions.
        """
        idx = (step - 1) % self.config.stride_steps
        if idx == 0:
            try:
                self.current_parts = self.strategy.partition(state)
                if self.pre_hook:
                    self.pre_hook(step, list(state.keys()))
                logger.debug(f"[Checkpoint] created {len(self.current_parts)} partitions at step {step}")
            except Exception as e:
                logger.error(f"Snapshot partition failed at step {step}: {e}")

    def persist(self, ckpt_id: int) -> None:
        """
        Persist one partition per step via AsyncWriter.
        After last partition, clear and invoke post_hook.
        """
        step = self.manager.current_step
        idx = (step - 1) % self.config.stride_steps
        if self.current_parts:
            part = self.current_parts[idx]
            self.writer.enqueue(ckpt_id, part)
            self.metrics[f"ckpt_{ckpt_id}_part_{idx}"] = len(part)
            if idx == self.config.stride_steps - 1:
                self.current_parts = []
                if self.post_hook:
                    self.post_hook(ckpt_id)
                logger.debug(f"[Checkpoint] completed cycle for ckpt {ckpt_id}")

    def aggregate_metrics(self) -> Dict[str, Any]:
        """
        Return collected metrics for monitoring.
        """
        return self.metrics.copy()

    def set_hooks(self,
                  pre_hook: Optional[Callable[[int, List[str]], None]] = None,
                  post_hook: Optional[Callable[[int], None]] = None) -> None:
        """
        Register custom hooks to run before snapshot and after cycle completion.
        """
        self.pre_hook = pre_hook
        self.post_hook = post_hook

    def shutdown(self) -> None:
        """
        Stop AsyncWriter and flush pending writes.
        """
        self.writer.stop()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass



class PartitionStrategy:
    """
    Strategy to split a model's state_dict into staggered partitions.
    """
    def __init__(self, stride_steps: int):
        self.stride_steps = stride_steps

    def partition(self, state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Split the state_dict into `stride_steps` partitions based on key order.
        """
        keys = sorted(state.keys())
        total = len(keys)
        base = total // self.stride_steps
        parts: List[Dict[str, torch.Tensor]] = []
        for i in range(self.stride_steps):
            start = i * base
            end = (i + 1) * base if i < self.stride_steps - 1 else total
            part_state = {k: state[k] for k in keys[start:end]}
            parts.append(part_state)
        return parts

class AsyncWriter(threading.Thread):
    """
    Background thread to persist checkpoint partitions asynchronously.
    """
    def __init__(self, save_dir: str):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.queue: Queue = Queue()
        self.running = True

    def run(self):
        while self.running or not self.queue.empty():
            try:
                ckpt_id, part = self.queue.get(timeout=1)
            except Empty:
                continue
            try:
                filename = f"weakckpt_{ckpt_id}.pt"
                path = os.path.join(self.save_dir, filename)
                torch.save(part, path)
                logger.info(f"[AsyncWriter] saved checkpoint {ckpt_id} ({len(part)} params)")
            except Exception as error:
                logger.error(f"[AsyncWriter] error saving {ckpt_id}: {error}")
            finally:
                self.queue.task_done()

    def enqueue(self, ckpt_id: int, part: Dict[str, torch.Tensor]):
        """Enqueue a partition for writing."""
        # clone tensors to avoid in-flight mutation
        safe_part = {k: v.clone().cpu() for k, v in part.items()}
        self.queue.put((ckpt_id, safe_part))
        logger.debug(f"[AsyncWriter] enqueued part for ckpt {ckpt_id}")

    def stop(self):
        """Stop the writer after flushing queue."""
        self.running = False
        self.queue.join()
        self.join()

class WeakCkptCheckpoint:
    """
    Handles staggered snapshot and asynchronous persistence of model parameters.
    """
    def __init__(self, manager, config):
        self.manager = manager
        self.config = config
        self.strategy = PartitionStrategy(config.stride_steps)
        self.current_parts: List[Dict[str, torch.Tensor]] = []
        self.writer = AsyncWriter(manager.save_dir)
        self.writer.start()

    def snapshot(self, state: Dict[str, torch.Tensor], step: int) -> None:
        """
        On the first step of each stride cycle, compute partitions.
        Subsequent steps reuse those partitions.
        """
        idx = (step - 1) % self.config.stride_steps
        if idx == 0:
            self.current_parts = self.strategy.partition(state)
            logger.debug(f"[Checkpoint] created {len(self.current_parts)} partitions")

    def persist(self, ckpt_id: int) -> None:
        """
        Persist one partition per step via AsyncWriter.
        """
        step = self.manager.current_step
        idx = (step - 1) % self.config.stride_steps
        if self.current_parts:
            part = self.current_parts[idx]
            self.writer.enqueue(ckpt_id, part)
            if idx == self.config.stride_steps - 1:
                # clear partitions after last one
                self.current_parts = []
                logger.debug(f"[Checkpoint] enqueued full cycle for ckpt {ckpt_id}")

    def shutdown(self) -> None:
        """
        Stop AsyncWriter and flush pending writes.
        """
        self.writer.stop()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
