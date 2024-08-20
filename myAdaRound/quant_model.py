import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import (
    quantizerDict,
    create_AdaRound_Quantizer,
    StraightThrough,
    NaiveDynnamicMinMaxQuantizer,
)
from .quant_layer import QuantLayer
from .quant_block import QuantBasicBlock

import torchvision.models.resnet as resnet

"""
conv1
bn1
relu
maxpool

layer2
layer2.0
layer2.0.conv1
layer2.0.bn1
layer2.0.relu
layer2.0.conv2
layer2.0.bn2
layer2.0.downsample
layer2.0.downsample.0
layer2.0.downsample.1
layer2.1
layer2.1.conv1
layer2.1.bn1
layer2.1.relu
layer2.1.conv2
layer2.1.bn2
...
avgpool
fc
"""


class QuantResNet(nn.Module):
    def __init__(self, orgModel, w_quant_args, a_quant_args, main_args):
        super(QuantResNet, self).__init__()
        args = main_args
        first_conv, first_bn = None, None

        self.ConvBnRelu1 = QuantLayer(
            conv_module=orgModel.conv1,
            bn_module=orgModel.bn1,
            act_module=nn.ReLU(),
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.Maxpool = orgModel.maxpool

        self.block0_0 = QuantBasicBlock(
            orgModel.layer1[0],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block0_1 = QuantBasicBlock(
            orgModel.layer1[1],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block1_0 = QuantBasicBlock(
            orgModel.layer2[0],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block1_1 = QuantBasicBlock(
            orgModel.layer2[1],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block2_0 = QuantBasicBlock(
            orgModel.layer3[0],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block2_1 = QuantBasicBlock(
            orgModel.layer3[1],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block3_0 = QuantBasicBlock(
            orgModel.layer4[0],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.block3_1 = QuantBasicBlock(
            orgModel.layer4[1],
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.Avgpool = orgModel.avgpool
        self.fc = orgModel.fc

        for name, module in self.named_modules():
            if hasattr(module, "a_quant_inited"):
                module.a_quant_inited = True
                module.a_quant_enable = True
                module.act_quantizer = NaiveDynnamicMinMaxQuantizer()
                print(f"{name} a_quant_inited")

    def forward(self, x):
        x = self.ConvBnRelu1(x)
        x = self.Maxpool(x)
        x = self.block0_0(x)
        x = self.block0_1(x)
        x = self.block1_0(x)
        x = self.block1_1(x)
        x = self.block2_0(x)
        x = self.block2_1(x)
        x = self.block3_0(x)
        x = self.block3_1(x)
        x = self.Avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
