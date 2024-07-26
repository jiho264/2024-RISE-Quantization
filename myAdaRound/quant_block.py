import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import quantizerDict, create_AdaRound_Quantizer, StraightThrough
from .quant_layer import QuantLayer


class QuantBasicBlock(nn.Module):

    def __init__(
        self,
        org_basicblock: nn.Module,
        w_quant_args: dict = None,
        a_quant_args: dict = None,
        folding: bool = False,
    ):
        super(QuantBasicBlock, self).__init__()

        self.relu = nn.ReLU()

        conv_1, bn_1 = None, None
        conv_2, bn_2 = None, None
        conv_d, bn_d = None, None

        for name, module in org_basicblock.named_children():
            if isinstance(module, (nn.Conv2d)):
                if name == "conv1":
                    conv_1 = module
                elif name == "conv2":
                    conv_2 = module
            elif isinstance(module, (nn.BatchNorm2d)):
                if name == "bn1":
                    bn_1 = module
                elif name == "bn2":
                    bn_2 = module

            elif isinstance(module, (nn.Sequential)):
                for dname, dmodule in module.named_children():
                    if dname == "0":
                        conv_d = dmodule
                    elif dname == "1":
                        bn_d = dmodule

        self.conv_bn_relu_1 = QuantLayer(
            conv_module=conv_1,
            bn_module=bn_1,
            act_module=nn.ReLU(),
            w_quant_args=w_quant_args,
            a_quant_args=a_quant_args,
            folding=folding,
        )
        self.conv_bn_2 = QuantLayer(
            conv_module=conv_2,
            bn_module=bn_2,
            w_quant_args=w_quant_args,
            a_quant_args=a_quant_args,
            folding=folding,
        )
        if conv_d != None and bn_d != None:
            self.conv_bn_down = QuantLayer(
                conv_module=conv_d,
                bn_module=bn_d,
                w_quant_args=w_quant_args,
                a_quant_args=a_quant_args,
                folding=folding,
            )
        else:
            self.conv_bn_down = None

    def forward(self, input: Tensor) -> Tensor:
        _identity = input.clone()
        _out = self.conv_bn_relu_1(input)
        _out = self.conv_bn_2(_out)

        if self.conv_bn_down != None:
            _identity = self.conv_bn_down(_identity)

        _out += _identity
        _out = self.relu(_out)
        return _out


# class QuantBottleneck(nn.Module):
#     def __init__(self):
#         super(QuantBottleneck, self).__init__()
#         raise NotImplementedError("QuantBottleneck is not implemented yet.")
