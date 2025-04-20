import torch
import deepspeed
from weakckpt.config import WeakCkptConfig
from weakckpt.deepspeed_plugin import WeakCkptDeepSpeedPlugin
from torchvision import datasets, transforms
from mymodel import MyModel


def train():
    # DeepSpeed 配置文件路径
    ds_config = 'ds_config.json'
    # 弱一致性检查点配置
    wc_config = WeakCkptConfig(stride_steps=4, max_version_diff=2, base_interval=200)

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 初始化 DeepSpeed
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=ds_config
    )

    # 创建 WeakCkpt 插件并绑定
    wc_plugin = WeakCkptDeepSpeedPlugin(wc_config, './weakckpt_ds')
    wc_plugin.attach_to_engine(engine)

    # 数据准备
    transform = transforms.Compose([...])
    dataset = datasets.ImageNet('/data/imagenet', transform=transform)
    loader = engine.train_dataloader(dataset)

    for step, batch in enumerate(loader):
        inputs, labels = batch
        loss = engine(inputs, labels)
        engine.backward(loss)
        engine.step()