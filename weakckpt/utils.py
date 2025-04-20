def compute_error_threshold(config, bandwidth: float) -> int:
    """
    动态计算 checkpoint 触发间隔：
      如果带宽低于阈值，则加快触发频率
    """
    base = config.base_interval
    if bandwidth < config.io_bw_threshold:
        return max(base // 2, 1)
    return base