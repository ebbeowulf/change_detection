import torch.nn as nn
import torchvision


def build_resnet(in_channels=1, arch='resnet18', pretrained=True):
    """ImageNet ResNet with the first conv adapted to in_channels and a single-logit head
    for binary classification (use with BCEWithLogitsLoss).
    Pretrained RGB filters are summed and spread evenly across the new input channels,
    so a 1-channel input sees the same filter response as a gray RGB image."""
    weights = 'DEFAULT' if pretrained else None
    model = getattr(torchvision.models, arch)(weights=weights)

    if in_channels != 3:
        old = model.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        w = old.weight.data.sum(dim=1, keepdim=True) / in_channels
        new.weight.data = w.repeat(1, in_channels, 1, 1)
        model.conv1 = new

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model