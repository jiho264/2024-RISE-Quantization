import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import *
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

        self.ConvBnRelu1 = QuantLayer(
            conv_module=orgModel.conv1,
            bn_module=orgModel.bn1,
            act_module=nn.ReLU(),
            w_quant_args=w_quant_args,
            folding=args["folding"],
        )
        self.ConvBnRelu1.weight = orgModel.conv1.weight.clone().detach()
        # self.ConvBnRelu1.bias = orgModel.conv1.bias.clone().detach() # ->> None

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

        self.fc = QuantLayer(
            conv_module=orgModel.fc, w_quant_args=w_quant_args, a_quant_args="fc"
        )
        self.fc.weight = orgModel.fc.weight.clone().detach()
        self.fc.bias = orgModel.fc.bias.clone().detach()
        self.input_act = QuantAct(16)

    def forward(self, x):
        with torch.device("cuda"):
            x, s_x = self.input_act(x)
            x, s_x = self.ConvBnRelu1(x, s_x)
            x = self.Maxpool(x)
            x, s_x = self.block0_0(x, s_x)
            x, s_x = self.block0_1(x, s_x)
            x, s_x = self.block1_0(x, s_x)
            x, s_x = self.block1_1(x, s_x)
            x, s_x = self.block2_0(x, s_x)
            x, s_x = self.block2_1(x, s_x)
            x, s_x = self.block3_0(x, s_x)
            x, s_x = self.block3_1(x, s_x)
            x = self.Avgpool(x)
            x = torch.flatten(x, 1)
            x, s_x = self.fc(x, s_x)

        return x


"""
W8A8 : 69.266%

"""
