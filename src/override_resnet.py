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
- [x] skip add에서 그냥 +를 nn.quantized.FloatFunctional()으로 바꾸기
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
        self.relu1 = self.relu
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.add1 = nn.quantized.FloatFunctional()
        self.act_obs = nn.quantized.FloatFunctional()
        self.zero = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.add1.add(x, identity)
        x = self.relu3(x)
        x = self.act_obs.add(
            x, self.zero
        )  # Activation 보려면 꼭 필요함. 근데 이거 있으면, convert 절대 불가함. convert 이후 backend에 해당 명령어가 없음.

        return x

    # old
    # def forward(self, x: Tensor) -> Tensor:
    #     identity = x

    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu1(x)

    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu2(x)

    #     x = self.conv3(x)
    #     x = self.bn3(x)

    #     if self.downsample is not None:
    #         identity = self.downsample(identity)

    #     x = self.add.add(x, identity)
    #     x = self.relu3(x)

    #     return x

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
        self.act_obs = nn.quantized.FloatFunctional()
        self.zero = torch.tensor(0.0)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.quant(x)
    #     x = super(ResNet_quan, self).forward(x)
    #     x = self.dequant(x)
    #     return x

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.act_obs.add(x, self.zero)  ## observer 넣으려고 만든 +0 연산

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x


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
