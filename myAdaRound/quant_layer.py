import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import Tensor
from .utils import *


class QuantLayer(nn.Module):
    def __init__(
        self,
        conv_module,
        bn_module=StraightThrough(),
        act_module=StraightThrough(),
        w_quant_args=None,
        a_quant_args=None,
        folding=False,
    ):
        super(QuantLayer, self).__init__()
        """forward function setting"""
        if isinstance(conv_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=conv_module.stride,
                padding=conv_module.padding,
                dilation=conv_module.dilation,
                groups=conv_module.groups,
            )
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        self.weight = conv_module.weight.clone().detach()

        if conv_module.bias != None:
            self.bias = conv_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(conv_module.weight.size(0)).to(
                conv_module.weight.device
            )

        self.act_func = act_module

        """Bn folding"""
        self.folding = folding
        # conv + bn
        if self.folding == True and bn_module != None:
            ## (1) My folding code / org_resnet18 : 69.758%
            _safe_std = torch.sqrt(bn_module.running_var + bn_module.eps)
            w_view = (conv_module.out_channels, 1, 1, 1)
            _gamma = bn_module.weight

            self.weight = self.weight * (_gamma / _safe_std).view(w_view)

            self.bias = (
                _gamma * (self.bias - bn_module.running_mean) / _safe_std
                + bn_module.bias
            )

            # print("    BN Folded!")

        self.IntLinearOperator = IntLinear(
            fwd_func=self.fwd_func,
            fwd_kwargs=self.fwd_kwargs,
            weight_fp32=self.weight,
            bias_fp32=self.bias,
        )

    def forward(self, x: Tensor, s_pre) -> Tensor:
        a_hat, s_a = self.IntLinearOperator(x, s_pre)

        a_hat = self.act_func(a_hat)

        return a_hat, s_a
