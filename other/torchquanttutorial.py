import torch, tqdm
import torch.nn as nn
from torch.quantization import prepare, convert

# Set up warnings
import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
warnings.filterwarnings(action="ignore", module=r"torch.ao.quantization")

# override the torchvision.models.resnet
from torchvision.models.resnet import (
    ResNet,
    ResNet50_Weights,
    Bottleneck,
    BasicBlock,
)
from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param

from torch.quantization import (
    MinMaxObserver,
    QConfig,
    default_per_channel_weight_observer,
)


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
        # print(f"    block input dtype   : {x.dtype}, {x.max()}")
        x = self.conv1(x)
        # print(f"    after conv1 dtype   : {x.dtype}, {x.max()}")
        x = self.bn1(x)
        # print(f"    after bn1 dtype     : {x.dtype}, {x.max()}")
        x = self.relu1(x)
        # print(f"    after relu1 dtype   : {x.dtype}, {x.max()}")

        x = self.conv2(x)
        # print(f"    after conv2 dtype   : {x.dtype}, {x.max()}")
        x = self.bn2(x)
        # print(f"    after bn2 dtype     : {x.dtype}, {x.max()}")
        x = self.relu2(x)
        # print(f"    after relu2 dtype   : {x.dtype}, {x.max()}")

        x = self.conv3(x)
        # print(f"    after conv3 dtype   : {x.dtype}, {x.max()}")
        x = self.bn3(x)
        # print(f"    after bn3 dtype     : {x.dtype}, {x.max()}")

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.add1.add(x, identity)
        # print(f"    after IdAdd dtype   : {x.dtype}, {x.max()}")
        x = self.relu3(x)
        # print(f"    after relu3 dtype   : {x.dtype}, {x.max()}")

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

    def forward(self, x: Tensor) -> Tensor:
        print("Inference...")
        print(f"input.dtype         : {x.dtype}, {x.max()}")
        x = self.quant(x)
        # print(f"after quant dtype   : {x.dtype}, {x.max()}")
        x = self.conv1(x)
        # print(f"after conv1 dtype   : {x.dtype}, {x.max()}")
        x = self.bn1(x)
        # print(f"after bn1 dtype     : {x.dtype}, {x.max()}")
        x = self.relu(x)
        # print(f"after relu dtype    : {x.dtype}, {x.max()}")
        x = self.maxpool(x)
        # print(f"after maxpool dtype : {x.dtype}, {x.max()}")

        x = self.layer1(x)
        # print(f"after layer1 dtype  : {x.dtype}, {x.max()}")
        x = self.layer2(x)
        # print(f"after layer2 dtype  : {x.dtype}, {x.max()}")
        x = self.layer3(x)
        # print(f"after layer3 dtype  : {x.dtype}, {x.max()}")
        x = self.layer4(x)
        print(f"after layer4 dtype  : {x.dtype}, {x.max()}")
        # print(x.__str__)
        if x.get_device() == -1:
            print(x.int_repr().max())
        # print(x.__repr__)
        # print(x.data.dtype)
        # print(x.size)
        # print(x.qscheme)
        # print(x.q_scale)
        # print(x.q_zero_point)

        x = self.avgpool(x)
        # print(f"after avgpool dtype : {x.dtype}, {x.max()}")
        x = torch.flatten(x, 1)
        # print(f"after flatten dtype : {x.dtype}, {x.max()}")
        x = self.fc(x)
        # print(f"after fc dtype      : {x.dtype}, {x.max()}")

        x = self.dequant(x)
        print(f"after dequant dtype : {x.dtype}, {x.max()}")
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


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


def GetDataset(batch_size=64):
    import torchvision
    import torchvision.transforms as transforms

    train_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    test_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device):
    model.eval().to(device)
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    cnt = 0
    with torch.no_grad():
        for image, target in tqdm.tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def main():
    # prepare the model
    _model = resnet50_quan(weights="IMAGENET1K_V1")
    _model.eval().to("cuda")

    _batch_size = 64

    train_loader, test_loader = GetDataset(batch_size=_batch_size)

    num_eval_batches = len(test_loader)

    # print("ㄴ Evaluation accuracy on 50k images, 76.13")
    # """
    #     # print("ㄴ Evaluation accuracy on 50k images, 76.13")
    #     train_dataset = torchvision.datasets.ImageNet(
    #     root="data/ImageNet",
    #     split="train",
    #     transform=transforms.Compose(
    #         [
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225],
    #             ),
    #         ]
    #     ),
    # )
    # """
    # top1, top5 = evaluate(
    #     _model, test_loader, neval_batches=num_eval_batches, device="cuda"
    # )
    # print("Original model evaluation...")
    # print(
    #     "Evaluation accuracy on %d images, %2.2f"
    #     % (num_eval_batches * _batch_size, top1.avg)
    # )

    # %%###############################################################################
    _model = fuse_ALL(_model)

    # _model.qconfig = torch.quantization.get_default_qconfig("x86")

    _model.qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, reduce_range=True),
        weight=default_per_channel_weight_observer.with_args(dtype=torch.qint8),
    )
    print(_model.qconfig)

    prepare(_model, inplace=True)

    # %% calibrate the model ############################################################
    calib_len = 16
    print("Calibrating the model...")
    print(f"ㄴ Complited with {_batch_size * calib_len} images...")
    for i, (data, _) in enumerate(train_loader):
        if i > calib_len:
            break
        with torch.no_grad():
            _model(data.to("cuda"))

    # %%convert the model ############################################################
    _model.to("cpu")
    convert(_model, inplace=True)

    # %%evaluate the model ############################################################

    print("Quantized model evaluation...")
    top1, top5 = evaluate(
        _model, test_loader, neval_batches=num_eval_batches, device="cpu"
    )
    print(
        "Evaluation accuracy on %d images, %2.2f"
        % (num_eval_batches * _batch_size, top1.avg)
    )


if __name__ == "__main__":
    print("ResNet50 quantization with pytorch tutorial...")
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    main()
