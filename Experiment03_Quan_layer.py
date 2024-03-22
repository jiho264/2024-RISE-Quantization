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
        case: Any,
        num_classes: int = 1000,
        weights: Optional[str] = None,
    ) -> None:
        super(ResNet_quan, self).__init__(block, layers, num_classes)
        if weights is not None:
            self.load_state_dict(torch.load(weights))
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.case = case

    def forward(self, x: Tensor) -> Tensor:
        print(self.case)
        # See note [TorchScript super()]
        if self.case == "conv1":
            x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.case == "layer1":
            x = self.quant(x)
        x = self.layer1(x)

        if self.case == "layer2":
            x = self.quant(x)
        x = self.layer2(x)

        if self.case == "layer3":
            x = self.quant(x)
        x = self.layer3(x)

        if self.case == "layer4":
            x = self.quant(x)
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
    case: str,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet_quan(block, layers, case, **kwargs)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def resnet50_quan(
    *,
    weights: Optional[ResNet50_Weights] = None,
    progress: bool = True,
    case,
    **kwargs: Any,
) -> ResNet:
    weights = ResNet50_Weights.verify(weights)
    return _resnet_quan(
        BottleNeck_quan, [3, 4, 6, 3], weights, progress, case, **kwargs
    )


###############################################################################################
def fuse_ALL(model) -> nn.Module:
    SingleTimeFlag = False
    for m in model.modules():
        if m.__class__.__name__ == ResNet_quan.__name__:
            if SingleTimeFlag == True:
                raise ValueError("ResNet_quan is already fused")
            SingleTimeFlag = True
            torch.quantization.fuse_modules(
                m,
                ["conv1", "bn1", "relu"],
                inplace=True,
            )

        if type(m) == BottleNeck_quan:

            torch.quantization.fuse_modules(
                m,
                [
                    ["conv1", "bn1", "relu1"],
                    ["conv2", "bn2", "relu2"],
                    ["conv3", "bn3"],
                ],
                inplace=True,
            )
            if m.downsample is not None:
                torch.quantization.fuse_modules(
                    m.downsample,
                    ["0", "1"],
                    inplace=True,
                )
    return model


###############################################################################################

import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *

"""
activation == HistogramObserver
"""
cases = [
    resnet50_quan(weights=pretrained_weights_mapping[50], case="conv1"),
    resnet50_quan(weights=pretrained_weights_mapping[50], case="layer1"),
    resnet50_quan(weights=pretrained_weights_mapping[50], case="layer2"),
    resnet50_quan(weights=pretrained_weights_mapping[50], case="layer3"),
    resnet50_quan(weights=pretrained_weights_mapping[50], case="layer4"),
]


for case in cases:
    # prepare the model

    _model = case
    _model.to("cpu")
    _model.eval()

    # set fuse ############################################################
    _model = fuse_ALL(_model)
    # set qconfig
    if _model.case == "conv1":
        pass
    elif _model.case == "layer1":
        _model.conv1.qconfig = None
    elif _model.case == "layer2":
        _model.conv1.qconfig = None
        _model.layer1.qconfig = None
    elif _model.case == "layer3":
        _model.conv1.qconfig = None
        _model.layer1.qconfig = None
        _model.layer2.qconfig = None
    elif _model.case == "layer4":
        _model.conv1.qconfig = None
        _model.layer1.qconfig = None
        _model.layer2.qconfig = None
        _model.layer3.qconfig = None

    _model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.HistogramObserver,
        weight=torch.quantization.default_per_channel_weight_observer,
    )
    prepare(_model, inplace=True)

    # calibrate the model ############################################################
    criterion = nn.CrossEntropyLoss()
    train_loader, _ = GetDataset(
        dataset_name="ImageNet",
        device="cuda",
        root="data",
        batch_size=256,
        num_workers=8,
    )
    _, _ = SingleEpochEval(_model, train_loader, criterion, "cuda", limit=2)

    # convert the model ############################################################
    _model.to("cpu")
    convert(_model, inplace=True)

    # evaluate the model ############################################################

    _batch_size = 32
    _, test_loader = GetDataset(
        dataset_name="ImageNet",
        device="cpu",
        root="data",
        batch_size=_batch_size,
        num_workers=8,
    )

    eval_loss, eval_acc = SingleEpochEval(
        model=_model, testloader=test_loader, criterion=criterion, device="cpu", limit=1
    )
    model_size = get_size_of_model(_model)
    inference_time = run_benchmark(_model, test_loader, "cpu", 10)
    print("------------------------------------------------------------")
    print(f"Case: {_model.case}")
    print(f"Model Size: {model_size}MB")
    print(f"Inference Time: {inference_time:.2f}ms")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Acc: {eval_acc:.3f}%")
    print("\n")

print("Done!")
