import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import Tensor
from .utils import quantizerDict, create_AdaRound_Quantizer, StraightThrough


class QuantModule(nn.Module):
    def __init__(
        self, org_module, w_quant_args, a_quant_args, bn_module=None, folding=False
    ):
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

        if org_module.bias != None:
            self.bias = org_module.bias.clone().detach()
        else:
            self.bias = torch.zeros(org_module.weight.size(0)).to(
                org_module.weight.device
            )

        self.act_func = StraightThrough()

        """Bn folding"""
        self.folding = folding
        # conv + bn
        if self.folding == True and bn_module != None:
            ## (1) My folding code / org_resnet18 : 69.758%
            _safe_std = torch.sqrt(bn_module.running_var + bn_module.eps)
            w_view = (org_module.out_channels, 1, 1, 1)
            _gamma = bn_module.weight

            self.weight = self.weight * (_gamma / _safe_std).view(w_view)

            self.bias = (
                _gamma * (self.bias - bn_module.running_mean) / _safe_std
                + bn_module.bias
            )

            ## (2) Origin bn folding code / org_resnet18: 69.758%
            # fold_bn_into_conv(org_module, bn_module)
            # self.weight = org_module.weight
            # if org_module.bias is not None:
            #     self.bias = org_module.bias
            self.bn_func = StraightThrough()
            print("    BN Folded!")
        elif self.folding == False and bn_module == None:
            # FC layer dose not have bn layer
            self.bn_func = StraightThrough()
        elif self.folding == False and bn_module != None:
            # conv and bn are not folded!!!
            self.bn_func = bn_module
        else:
            raise ValueError("Unknown folding option")

        """weight quantizer"""
        # default is True. Need false option when only compute adaround values.
        self.w_quant_enable = True

        try:
            if w_quant_args.get("AdaRound") == True:
                self.weight_quantizer = create_AdaRound_Quantizer(
                    scheme=w_quant_args.get("scheme"),
                    org_weight=self.weight,
                    args=w_quant_args,
                )
            else:
                self.weight_quantizer = quantizerDict[w_quant_args.get("scheme")](
                    org_weight=self.weight, args=w_quant_args
                )
        except KeyError:
            raise ValueError(f"Unknown quantizer type: {w_quant_args.get('scheme')}")

        """activation quantizer"""
        if a_quant_args == {}:
            self.a_quant_enable = False
            self.a_quant_inited = False
        else:
            self.a_quant_enable = True
            self.a_quant_inited = False
            self.a_quant_args = a_quant_args
            self.act_quantizer = None

    def init_act_quantizer(self, calib):
        try:
            self.act_quantizer = quantizerDict[self.a_quant_args.get("scheme")](
                org_weight=calib, args=self.a_quant_args
            )
            self.act_quantizer._scaler = nn.Parameter(
                self.act_quantizer._scaler, requires_grad=True
            )
        except KeyError:
            raise ValueError(
                f"Unknown quantizer type: {self.a_quant_args.get('scheme')}"
            )

    def forward(self, x: Tensor) -> Tensor:
        """convolution"""
        if self.w_quant_enable == True:
            # print("q", end="")
            weight = self.weight_quantizer(self.weight)
        else:
            print(".", end="")
            weight = self.weight
        _Z = self.fwd_func(x, weight, self.bias, **self.fwd_kwargs)

        """ batch normalization """
        _Z = self.bn_func(_Z)

        """ activation """
        # If first conv of first block of each stage, it is ReLU.
        # Otherwise, it is StraightThrough.
        _A = self.act_func(_Z)

        if self.a_quant_inited == True and self.a_quant_enable == True:
            # print("A", end="")
            return self.act_quantizer(_A)
        else:
            return _A
