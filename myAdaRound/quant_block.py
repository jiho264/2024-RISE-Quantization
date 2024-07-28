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

        self.w_quant_enable = True
        self.a_quant_enable = False
        self.a_quant_inited = False

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

    def init_act_quantizer(self, calib):
        self.conv_bn_relu_1.init_act_quantizer(calib)

        self.conv_bn_2.init_act_quantizer(calib)

        if self.conv_bn_down != None:
            self.conv_bn_down.init_act_quantizer(calib)

        self.a_quant_inited = True

        print("Activation quantizer initialized from QuantBasicBlock")

    def get_rounding_parameter(self):
        _list = [
            self.conv_bn_relu_1.weight_quantizer._v,
            self.conv_bn_2.weight_quantizer._v,
        ]
        if self.conv_bn_down != None:
            _list.append(self.conv_bn_down.weight_quantizer._v)
        return _list

    def get_scaler_parameter(self):
        _list = [
            self.conv_bn_relu_1.act_quantizer._scaler,
            self.conv_bn_2.act_quantizer._scaler,
        ]
        if self.conv_bn_down != None:
            _list += self.conv_bn_down.act_quantizer._scaler
        return _list

    def get_sum_of_f_reg_with_lambda(self, beta):
        _sum = (
            self.conv_bn_relu_1.weight_quantizer.lamda
            * self.conv_bn_relu_1.weight_quantizer.f_reg(beta=beta)
        )
        _sum += (
            self.conv_bn_2.weight_quantizer.lamda
            * self.conv_bn_2.weight_quantizer.f_reg(beta=beta)
        )
        if self.conv_bn_down != None:
            _sum += (
                self.conv_bn_down.weight_quantizer.lamda
                * self.conv_bn_down.weight_quantizer.f_reg(beta=beta)
            )

        return _sum

    def setRoundingValues(self):
        self.conv_bn_relu_1.weight_quantizer.setRoundingValues()
        self.conv_bn_2.weight_quantizer.setRoundingValues()

        if self.conv_bn_down != None:
            self.conv_bn_down.weight_quantizer.setRoundingValues()

    def _quant_enabler(self):
        self.conv_bn_relu_1.w_quant_enable = self.w_quant_enable
        self.conv_bn_relu_1.a_quant_enable = self.a_quant_enable

        self.conv_bn_2.w_quant_enable = self.w_quant_enable
        self.conv_bn_2.a_quant_enable = self.a_quant_enable

        if self.conv_bn_down != None:
            self.conv_bn_down.w_quant_enable = self.w_quant_enable
            self.conv_bn_down.a_quant_enable = self.a_quant_enable

    def forward(self, input: Tensor) -> Tensor:
        self._quant_enabler()

        _identity = input
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
