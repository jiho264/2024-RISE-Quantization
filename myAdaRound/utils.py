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
    def __init__(self, org_weight, args):
        super(UniformAffineQuantizer, self).__init__()

        _DtypeStr = args.get("dstDtype")
        assert _DtypeStr in [
            "INT8",
            "UINT8",
            "INT4",
            "UINT4",
        ], f"Unknown quantization type: {_DtypeStr}"

        self.n_bits = int(_DtypeStr[-1])  # "INT8" -> 8
        self.signed = True if _DtypeStr[0] == "I" else False  # "INT8" -> True

        # the below code runtime is 300ms in resnet18 with i7-9700k with RTX3090
        if self.signed == False and org_weight.min() < 0:
            raise ValueError("Unsigned quantization does not support negative values.")

        self.repr_min, self.repr_max = None, None
        if self.signed:
            self.repr_min = -(2 ** (self.n_bits - 1))  # "INT8" -> -128
            self.repr_max = 2 ** (self.n_bits - 1) - 1  # "INT8" -> 127
        else:
            self.repr_min = 0  # "UINT8" -> 0
            self.repr_max = 2 ** (self.n_bits) - 1  # "UINT8" -> 255

        # per_ch option is "store_true
        self.per_channel = args.get("per_channel") if args.get("per_channel") else False
        self.n_ch = len(org_weight.size()) if self.per_channel else 1
        self.scaler = None
        self.zero_point = None

    def compute_qparams(self, _min, _max):
        # Origin tensor shape: [out_channel, in_channel, k, k]
        # per_ch Qparam shape: [n_channel, 1, 1, 1]

        scaler = (_max - _min) / (self.repr_max - self.repr_min)
        self.scaler = scaler.view(-1, *([1] * (self.n_ch - 1)))

        _min = _min.view(-1, *([1] * (self.n_ch - 1)))
        self.zero_point = -(_min / self.scaler).round() + self.repr_min

    def quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            (input / self.scaler).round() + self.zero_point,
            self.repr_min,
            self.repr_max,
        )

    def dequantize(self, input: Tensor) -> Tensor:
        return (input - self.zero_point) * self.scaler

    def forward(self, x: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x))


class AbsMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        [ Absolute Maximum Quantization ]
        - When zero_point is zero, the quantization range is symmetric.
            (Uniform Symmetric Quantization)
        - range: [-max(abs(x)), max(abs(x))]

        [W8A32]
        # resnet18 Acc@1 per-layer : 69.54 %
        # resnet18 Acc@1 per-channel : 69.64 %
        """
        super(AbsMaxQuantizer, self).__init__(org_weight, args)

        _AbsMax = None
        if self.per_channel == True:
            _AbsMax = org_weight.view(org_weight.size(0), -1).abs().max(dim=1).values
        else:
            _AbsMax = org_weight.abs().max()

        # if s8, scaler = 2 * org_weight.abs().max() / (127 - (-128))
        # if u8, scaler = 2 * org_weight.abs().max() / (255 - 0)

        # if s8 or u8, zero_point = 0
        self.compute_qparams(-_AbsMax, _AbsMax)
        self.zero_point = torch.zeros_like(self.scaler)

        # for debug
        # print(self.scaler.shape, self.zero_point.shape)


class MinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        [ Min-Max Quantization ]
        - The quantization range is asymmetric.
        - range: [min(x), max(x)]
        - https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver

        [W8A32]
        # resnet18 Acc@1 per-layer : 69.65 %
        # resnet18 Acc@1 per-channel : 69.76 % (same with origin)
        """
        super(MinMaxQuantizer, self).__init__(org_weight, args)

        if self.per_channel == True:
            _min = org_weight.view(org_weight.size(0), -1).min(dim=1).values
            _max = org_weight.view(org_weight.size(0), -1).max(dim=1).values
        else:
            _min = org_weight.min()
            _max = org_weight.max()

        # if s8, scaler = (_max - _min) / (127 - (-128))
        # if u8, scaler = (_max - _min) / (255 - 0)

        # if s8, zero_point = -_min / scaler + (-128)
        # if u8, zero_point = -_min / scaler + 0
        self.compute_qparams(_min, _max)

        # for debug
        # print(self.scaler.shape, self.zero_point.shape)


class L2DistanceQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        Args:
            args (dict): for UniformAffineQuantizer
            org_weight (Tensor): have to need for determine the scaling factor
        """
        super(L2DistanceQuantizer, self).__init__(org_weight, args)

        # [ ] implement the L2DistanceQuantizer
        """below is uncomplited"""
        if self.per_channel == True:
            _min = org_weight.view(org_weight.size(0), -1).min(dim=1).values
            _max = org_weight.view(org_weight.size(0), -1).max(dim=1).values
        else:
            _min = org_weight.min()
            _max = org_weight.max()

        if self.signed == True:
            # perform_2D_search [min, max]
            best_l2_norm = torch.ones_like(_min) * 9999
            best_scaler = torch.ones_like(_min)
            best_zero_point = torch.zeros_like(_min)

            for i in range(0, 99):
                _tmp_min = _min * (100 - i) / 100  # 100%, 99%, 98%, ..., 1%
                _tmp_max = _max * (100 - i) / 100  # 100%, 99%, 98%, ..., 1%

                self.scaler = (_tmp_max - _tmp_min) / (self.repr_max - self.repr_min)
                self.zero_point = -(_tmp_min / self.scaler).round() + self.repr_min
                # per-ch인 경우 l2 score가 ch수 만큼 생성되어야함.
                # 그 뒤 각 ch마다 점수 좋은 것을 선택해야함.

                # 여기 p=2말고 p=2.4인 경우 더 좋은 결과를 보임.
                _l2_norm = (self.forward(org_weight) - org_weight).norm(p=2)

                # best_l2_norm = torch.where(_l2_norm < best_l2_norm,

                if _l2_norm < best_l2_norm:
                    best_l2_norm = _l2_norm
                    best_scaler = self.scaler
                    best_zero_point = self.zero_point

            self.scaler = best_scaler
            self.zero_point = best_zero_point
        else:
            # perform_1D_search [0, max]
            # best_l2_norm = 9999
            # best_scaler = 1.0
            # best_zero_point = 0.0

            # for i in range(0, 80):
            #     _tmp_min = _min * 0.01 * i
            #     _tmp_max = _max * 0.01 * i

            #     self.scaler = (_tmp_max - _tmp_min) / (self.repr_max - self.repr_min)
            #     self.zero_point = -(_tmp_min / self.scaler).round() + self.repr_min

            #     _l2_norm = (self.forward(org_weight) - org_weight).norm(p=2)
            #     if _l2_norm < best_l2_norm:
            #         best_l2_norm = _l2_norm
            #         best_scaler = self.scaler
            #         best_zero_point = self.zero_point

            # self.scaler = best_scaler
            # self.zero_point = best_zero_point
            ...


class QuantModule(nn.Module):
    def __init__(self, org_module, weight_quant_params, act_quant_params):
        super(QuantModule, self).__init__()
        """forward function setting"""
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
        w_quantizer_type = weight_quant_params.get("scheme")

        if w_quantizer_type == "AbsMaxQuantizer":
            self.weight_quantizer = AbsMaxQuantizer(self.weight, weight_quant_params)

        elif w_quantizer_type == "MinMaxQuantizer":
            self.weight_quantizer = MinMaxQuantizer(self.weight, weight_quant_params)
        elif w_quantizer_type == "L2DistanceQuantizer":
            self.weight_quantizer = L2DistanceQuantizer(
                self.weight, weight_quant_params
            )

            # [ ] add more quantizer option

            pass
        else:
            raise ValueError(f"Unknown weight quantizer type: {w_quantizer_type}")

        """activation quantizer"""
        # [ ] add activation quantizer

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight_quantizer(self.weight)
        # print(".", end="")

        return self.fwd_func(x, weight, **self.fwd_kwargs)
