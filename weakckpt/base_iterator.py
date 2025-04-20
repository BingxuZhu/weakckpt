from collections import deque

class WeakCkptBaseIterator:
    """
    可恢复迭代器基类、
    保存最新索引以支持断点续训
    """
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0
        self.history = deque(maxlen=5)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch = self.dataset[self.index: self.index + self.batch_size]
        self.history.append(self.index)
        self.index += self.batch_size
        return batch

    def get_state(self) -> dict:
        """返回迭代器当前状态，用于 checkpoint snapshot"""
        return {
            'index': self.index,
            'history': list(self.history),
        }