from collections import deque
import torch
import os
import json
import threading
from typing import Any, Dict, List, Optional

class WeakCkptBaseIterator:
    """
    Recoverable iterator base class:
    - Maintains progress index and history
    - Supports saving and loading iterator state for checkpointing
    """
    def __init__(self, dataset: List[Any], batch_size: int = 1, state_file: Optional[str] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.history = deque(maxlen=10)
        self.state_file = state_file or 'weakckpt_iter_state.json'
        self._lock = threading.Lock()
        self._load_state()

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        with self._lock:
            if self.index >= len(self.dataset):
                raise StopIteration
            batch = self.dataset[self.index: self.index + self.batch_size]
            self.history.append(self.index)
            self.index += self.batch_size
            return batch

    def get_state(self) -> Dict[str, Any]:
        """
        Return current iterator state for checkpointing:
        { 'index': int, 'history': List[int] }
        """
        return {
            'index': self.index,
            'history': list(self.history)
        }

    def save_state(self) -> None:
        """
        Persist iterator state to JSON file.
        """
        state = self.get_state()
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def _load_state(self) -> None:
        """
        Load iterator state from JSON if available.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.index = state.get('index', 0)
                self.history = deque(state.get('history', []), maxlen=10)
            except Exception:
                self.index = 0
                self.history.clear()

    def checkpoint(self) -> None:
        """
        Alias for save_state to integrate with manager.
        """
        self.save_state()

    def rewind(self, steps: int) -> None:
        """
        Move iterator backward by given number of steps, useful for error recovery.
        """
        with self._lock:
            target = max(0, self.index - steps * self.batch_size)
            self.index = target
            while self.history and self.history[-1] >= self.index:
                self.history.pop()

    def __len__(self) -> int:
        return len(self.dataset)

class WeakCkptIterator(WeakCkptBaseIterator):
    """
    Iterator that integrates with WeakCkptManager for automatic checkpointing.
    """
    def __init__(self, dataset: List[Any], manager, batch_size: int = 1):
        super().__init__(dataset, batch_size)
        self.manager = manager

    def __next__(self) -> Any:
        batch = super().__next__()
        state = self.get_state()
        # take checkpoint of iterator state
        self.manager.ckpt.snapshot({'iterator_state': state}, self.manager.current_step)
        return batch