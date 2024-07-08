import torch, tqdm
from torch import Tensor


#################################################################################################
## 1. Prepare the dataset and utility functions
#################################################################################################
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
        shuffle=False,
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
        # for image, target in tqdm.tqdm(data_loader):
        for image, target in data_loader:
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


#################################################################################################
## 2. Quantized module
#################################################################################################

import torch.nn as nn
import torch.nn.functional as F


class UniformAffineQuantizer(nn.Module):
    def __init__(self, args):
        """QuantizerBase
            - Base class for quantizer
            - Uniform Affine Quantization

        Args:
            args (_type_): dict
                - active (bool): quantization enabled or not
                - n_bits (int): number of bits for quantization
                - per_channel (bool): per channel quantization or not
        """
        super(UniformAffineQuantizer, self).__init__()
        self.n_bits = args.get("n_bits")
        self.n_steps = 2**self.n_bits
        self.per_channel = args.get("per_channel")
        self.scaler = 1.0
        self.zero_point = 0

    def quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            (input / self.scaler).round() + self.zero_point,
            -self.n_steps // 2,
            self.n_steps // 2 - 1,
        )

    def dequantize(self, input: Tensor) -> Tensor:
        return (input - self.zero_point) * self.scaler

    def forward(self, x: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x))


class AbsMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, args, org_weight=None):
        """AbsMinMaxQuantizer.
        Naive symmetric quantizer with abs max scaling.
        Args:
            args (dict): for UniformAffineQuantizer
            org_weight (Tensor): have to need for determine the scaling factor
        """
        super(AbsMaxQuantizer, self).__init__(args)

        if self.per_channel == True:

            self.scaler = org_weight.view(org_weight.size(0), -1).abs().max(
                dim=1
            ).values / (self.n_steps // 2 - 1)
            # print(self.scaler.shape)
            self.scaler = self.scaler.view(-1, *([1] * (len(org_weight.size()) - 1)))
            # print(org_weight.shape, self.scaler.shape)
        else:
            self.scaler = org_weight.abs().max() / (self.n_steps // 2 - 1)
            print(org_weight.shape, self.scaler.shape)


class MinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, args, org_weight=None):
        """MinMaxQuantizer.

        Args:
            args (dict): for UniformAffineQuantizer
            org_weight (Tensor): have to need for determine the scaling factor
        """
        super(MinMaxQuantizer, self).__init__(args)

        if self.per_channel == True:
            _mins = org_weight.view(org_weight.size(0), -1).min(dim=1).values
            _maxs = org_weight.view(org_weight.size(0), -1).max(dim=1).values

            self.scaler = (_maxs - _mins) / (self.n_steps // 2 - 1)
            self.scaler = self.scaler.view(-1, *([1] * (len(org_weight.size()) - 1)))
        else:
            a = org_weight.min()
            b = org_weight.max()

            self.scaler = (b - a) / (self.n_steps - 1)


class QuantModule(nn.Module):

    def __init__(self, org_module, weight_quant_params, act_quant_params):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.weight = org_module.weight.clone().detach()

        """weight quantizer"""
        self.w_act = weight_quant_params.get("active")
        w_quantizer_type = weight_quant_params.get("quantizer")

        if w_quantizer_type == "AbsMaxQuantizer":
            self.weight_quantizer = AbsMaxQuantizer(weight_quant_params, self.weight)

        elif w_quantizer_type == "MinMaxQuantizer":
            self.weight_quantizer = MinMaxQuantizer(weight_quant_params, self.weight)

            # [ ] add more quantizer option
            pass
        else:
            raise ValueError(f"Unknown weight quantizer type: {w_quantizer_type}")

    def forward(self, x: Tensor) -> Tensor:
        if self.w_act == True:
            weight = self.weight_quantizer(self.weight)
            print(".", end="")
        else:
            weight = self.weight

        return self.fwd_func(x, weight, **self.fwd_kwargs)

        """
        ResNet18 quantization with myAdaRound...
        QuantModule: conv1, torch.Size([64, 3, 7, 7])
        QuantModule: layer1.0.conv1, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.0.conv2, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.1.conv1, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.1.conv2, torch.Size([64, 64, 3, 3])
        QuantModule: layer2.0.conv1, torch.Size([128, 64, 3, 3])
        QuantModule: layer2.0.conv2, torch.Size([128, 128, 3, 3])
        QuantModule: layer2.0.downsample.0, torch.Size([128, 64, 1, 1])
        QuantModule: layer2.1.conv1, torch.Size([128, 128, 3, 3])
        QuantModule: layer2.1.conv2, torch.Size([128, 128, 3, 3])
        QuantModule: layer3.0.conv1, torch.Size([256, 128, 3, 3])
        QuantModule: layer3.0.conv2, torch.Size([256, 256, 3, 3])
        QuantModule: layer3.0.downsample.0, torch.Size([256, 128, 1, 1])
        QuantModule: layer3.1.conv1, torch.Size([256, 256, 3, 3])
        QuantModule: layer3.1.conv2, torch.Size([256, 256, 3, 3])
        QuantModule: layer4.0.conv1, torch.Size([512, 256, 3, 3])
        QuantModule: layer4.0.conv2, torch.Size([512, 512, 3, 3])
        QuantModule: layer4.0.downsample.0, torch.Size([512, 256, 1, 1])
        QuantModule: layer4.1.conv1, torch.Size([512, 512, 3, 3])
        QuantModule: layer4.1.conv2, torch.Size([512, 512, 3, 3])
        QuantModule: fc, torch.Size([1000, 512])
        """
