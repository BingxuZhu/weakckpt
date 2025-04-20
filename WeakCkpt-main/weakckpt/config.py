class WeakCkptConfig:
    """
    弱一致性检查点配置
    stride_steps: 跨步数量
    max_version_diff: 最大允许的版本差
    base_interval: 基准触发间隔（训练步数）
    io_bw_threshold: I/O 带宽阈值
    """
    def __init__(
        self,
        stride_steps: int = 2,
        max_version_diff: int = 1,
        base_interval: int = 100,
        io_bw_threshold: float = 100.0,
    ):
        self.stride_steps = stride_steps
        self.max_version_diff = max_version_diff
        self.base_interval = base_interval
        self.io_bw_threshold = io_bw_threshold