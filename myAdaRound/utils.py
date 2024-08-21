import torch.nn as nn
import torch.nn.functional as F

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, EncoderBlock, MLPBlock
from collections import OrderedDict
from torch import Tensor
from typing import Optional, Tuple

""" Quantizer """

import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time

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
        for image, target in tqdm.tqdm(data_loader):
            # for image, target in data_loader:
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


class QuantAct(nn.Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'asymmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self, activation_bit=16):
        super(QuantAct, self).__init__()
        # print("qant!")
        self.activation_bit = activation_bit

    def forward(self, x_hat, s_x=None, id_hat=None, s_id=None):
        with torch.no_grad():
            x_out = x_hat if id_hat == None else id_hat + x_hat

            s_out = x_out.abs().max() / (2 ** (self.activation_bit - 1) - 1)

        if s_x == None:
            # input quantization
            out_int = (
                (x_hat / s_out)
                .round()
                .clamp(
                    -(2 ** (self.activation_bit - 1)),
                    2 ** (self.activation_bit - 1) - 1,
                )
            )
        else:
            x_int = (x_hat / s_x).round()
            new_scaler = s_x / s_out
            out_int = (
                (x_int * new_scaler)
                .round()
                .clamp(
                    -(2 ** (self.activation_bit - 1)),
                    2 ** (self.activation_bit - 1) - 1,
                )
            )

            if id_hat is not None:
                id_int = (id_hat / s_id).round()
                new_scaler = s_id / s_out
                id_int = (
                    (id_int * new_scaler)
                    .round()
                    .clamp(
                        -(2 ** (self.activation_bit - 1)),
                        2 ** (self.activation_bit - 1) - 1,
                    )
                )

                out_int += id_int

        return out_int * s_out.view(-1), s_out


class IntLinear(nn.Module):
    def __init__(self, fwd_func, fwd_kwargs, weight_fp32, bias_fp32, bits=16):
        """INT8 GEMM"""
        super(IntLinear, self).__init__()
        self.fwd_func = fwd_func
        self.fwd_kwargs = fwd_kwargs

        self.bits = bits

        # only per-channel quantization is supported
        s_w = weight_fp32.view(weight_fp32.size(0), -1).abs().max(dim=1).values / (
            2 ** (bits - 1) - 1
        )

        self._n_ch = len(weight_fp32.size())
        self.s_w = s_w.view(-1, *([1] * (self._n_ch - 1)))

        self.w_int8 = (
            (weight_fp32.clone().detach() / self.s_w)
            .round()
            .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        )
        self.b_int8 = (bias_fp32.clone().detach() / self.s_w.squeeze()).round()

        self.qact = QuantAct(bits)

    def forward(self, x_hat, s_x):
        # 여기 x의 bit repr는 이전 layer에서 몇으로 줄였느냐에 따라 다름
        x_int = (x_hat / s_x).round()

        s_a = self.s_w * s_x
        if self.s_w.dim() == 4:
            s_a = s_a.view(1, -1, 1, 1)
        elif self.s_w.dim() == 2:
            s_a = s_a.view(1, -1)

        b_int32 = (self.b_int8 / s_x).round()

        a_int32 = self.fwd_func(x_int, self.w_int8, b_int32, **self.fwd_kwargs)
        a_hat = a_int32 * s_a

        return self.qact(a_hat, s_a)


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input
