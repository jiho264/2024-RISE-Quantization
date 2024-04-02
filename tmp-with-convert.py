import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *

# from src.override_resnet import *
# from src.custom_observer import CustomHistogramObserver


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
        self.through = nn.quantized.FloatFunctional()
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
            # print(self.downsample[0].stride)

        x = self.add1.add(x, identity)
        x = self.relu3(x)
        # x = self.through.add(
        #     x, self.zero
        # )  # Activation 보려면 꼭 필요함. 근데 이거 있으면, convert 절대 불가함. convert 이후 backend에 해당 명령어가 없음.

        return x


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
        # self.act_obs = nn.quantized.FloatFunctional()
        self.zero = torch.tensor(0.0)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.quant(x)
    #     x = super(ResNet_quan, self).forward(x)
    #     x = self.dequant(x)
    #     return x

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.act_obs.add(x, self.zero)  ## observer 넣으려고 만든 +0 연산
        x = self.quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dequant(x)

        x = self.fc(x)

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


# for case_activation in cases_activation:
#     for case_weight in cases_weight:
# prepare the model
# for i in [120, 121, 122, 123, 124, 125, 126, 127, 128]:
# for i in range(110, 121):
_model = resnet50_quan(weights=pretrained_weights_mapping[50])
_model.to("cpu")
_model.eval()

# set fuse ############################################################
_model = fuse_ALL(_model)

_model.conv1.qconfig = None
_model.fc.qconfig = None
_model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.HistogramObserver.with_args(
        quant_max=127, quant_min=0, dtype=torch.quint8
    ),
    weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
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
_, _ = SingleEpochEval(_model, train_loader, criterion, "cuda", 4)

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
    model=_model,
    testloader=test_loader,
    criterion=criterion,
    device="cpu",
    limit=10,
)
model_size = get_size_of_model(_model)
inference_time = run_benchmark(_model, test_loader, "cpu", 10)
print("------------------------------------------------------------")
# print(f"case_activation: {case_activation}")
# print(f"case_upsample_rate: {i}")
print(f"Model Size: {model_size:.2f}MB")
print(f"Inference Time: {inference_time:.2f}ms")
print(f"Eval Loss: {eval_loss:.4f}")
print(f"Eval Acc: {eval_acc:.3f}%")
print("\n")

print("Done!")


"""

first conv, last fc 제외하고 싹다 quantization 때리는 코드


"""
