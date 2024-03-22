# %% override the torchvision.models.resnet
from torchvision.models.resnet import (
    ResNet,
    ResNet50_Weights,
    Bottleneck,
    BasicBlock,
)
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

"""
Todo : 
- [x] forward 함수 앞뒤로 quantization 추가
- [ ] skip add에서 그냥 +를 nn.quantized.FloatFunctional()으로 바꾸기
- [x] Conv, bn, relu 하나로 만들어야함.
- [x] ReLU 6면 int계산 안 되는데, 일반 ReLU인 것은 확인 완료
"""


class BottleNeck_quan(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BottleNeck_quan, self).__init__(
            inplanes,
            planes,
            stride,
            downsample,
            groups,
            base_width,
            dilation,
            norm_layer,
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.add.add(out, identity)
        out = self.relu3(out)

        return out

    # def forward(self, x: Tensor) -> Tensor:
    #     x = super(BottleNeck_quan, self).forward(x)
    #     return x


class ResNet_quan(ResNet):
    def __init__(
        self,
        block: Any,
        layers: list[int],
        num_classes: int = 1000,
        weights: Optional[str] = None,
    ) -> None:
        super(ResNet_quan, self).__init__(block, layers, num_classes)
        if weights is not None:
            self.load_state_dict(torch.load(weights))
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = super(ResNet_quan, self).forward(x)
        x = self.dequant(x)
        return x

    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x

    # def forward(self, x: Tensor) -> Tensor:
    #     return self._forward_impl(x)


def _resnet_quan(
    block: Type[Union[BasicBlock, BottleNeck_quan]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_quan(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def resnet50_quan(
    *, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    weights = ResNet50_Weights.verify(weights)
    return _resnet_quan(BottleNeck_quan, [3, 4, 6, 3], weights, progress, **kwargs)
