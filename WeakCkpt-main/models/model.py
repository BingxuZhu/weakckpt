import torch
from weakckpt.config import WeakCkptConfig
from weakckpt.manager import WeakCkptManager
from torchvision import datasets, transforms


def train():
    config = WeakCkptConfig(stride_steps=4, max_version_diff=2, base_interval=200)
    manager = WeakCkptManager('./ckpts', config)
    transform = transforms.Compose([...])
    dataset = datasets.ImageNet('/data/imagenet', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    model = MyModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        state = model.state_dict()
        manager.on_step_end(state)