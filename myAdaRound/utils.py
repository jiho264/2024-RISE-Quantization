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
    ## Public variables :
    # - (bool) signed
    # - (bool) per-channel
    ## Public methods :
    # - compute_qparams(_min, _max)
    # - forward(x: Tensor) -> Tensor
    def __init__(self, org_weight, args):
        super(UniformAffineQuantizer, self).__init__()

        _DtypeStr = args.get("dstDtype")
        assert _DtypeStr in [
            "INT8",
            "UINT8",
            "INT4",
            "UINT4",
        ], f"Unknown quantization type: {_DtypeStr}"

        self._n_bits = int(_DtypeStr[-1])  # "INT8" -> 8
        self.signed = True if _DtypeStr[0] == "I" else False  # "INT8" -> True

        # the below code runtime is 300ms in resnet18 with i7-9700k with RTX3090
        if self.signed == False and org_weight.min() < 0:
            raise ValueError("Unsigned quantization does not support negative values.")

        self._repr_min, self._repr_max = None, None
        if self.signed:
            self._repr_min = -(2 ** (self._n_bits - 1))  # "INT8" -> -128
            self._repr_max = 2 ** (self._n_bits - 1) - 1  # "INT8" -> 127
        else:
            self._repr_min = 0  # "UINT8" -> 0
            self._repr_max = 2 ** (self._n_bits) - 1  # "UINT8" -> 255

        # per_ch option is "store_true"
        self.per_channel = args.get("per_channel") if args.get("per_channel") else False
        self._n_ch = len(org_weight.size()) if self.per_channel else 1
        self._scaler = None
        self._zero_point = None

    def compute_qparams(self, _min, _max) -> None:
        # Origin tensor shape: [out_channel, in_channel, k, k]
        # per_ch Qparam shape: [n_channel, 1, 1, 1]

        scaler = (_max - _min) / (self._repr_max - self._repr_min)
        self._scaler = scaler.view(-1, *([1] * (self._n_ch - 1)))

        _min = _min.view(-1, *([1] * (self._n_ch - 1)))
        self._zero_point = -(_min / self._scaler).round() + self._repr_min

    def _quantize(self, input: Tensor) -> Tensor:
        return torch.clamp(
            (input / self._scaler).round() + self._zero_point,
            self._repr_min,
            self._repr_max,
        )

    def _dequantize(self, input: Tensor) -> Tensor:
        return (input - self._zero_point) * self._scaler

    def forward(self, x: Tensor) -> Tensor:
        return self._dequantize(self._quantize(x))


class AbsMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        [ Absolute Maximum Quantization ]
        - When zero_point is zero, the quantization range is symmetric.
            (Uniform Symmetric Quantization)
        - range: [-max(abs(x)), max(abs(x))]
        """
        super(AbsMaxQuantizer, self).__init__(org_weight, args)

        _AbsMax = None
        if self.per_channel == True:
            _AbsMax = org_weight.view(org_weight.size(0), -1).abs().max(dim=1).values
        else:
            _AbsMax = org_weight.abs().max()

        if self.signed == True:
            # if s8, scaler = 2 * org_weight.abs().max() / (127 - (-128))
            #               = org_weight.abs().max() - (-org_weight.abs().max()) / (127 - (-128))
            self.compute_qparams(-_AbsMax, _AbsMax)
        else:
            # if u8, scaler = org_weight.abs().max() / (255 - 0)
            self.compute_qparams(torch.zeros_like(_AbsMax), _AbsMax)

        # if s8 or u8, zero_point = 0
        self.zero_point = torch.zeros_like(self._scaler)


class MinMaxQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        [ Min-Max Quantization ]
        - The quantization range is asymmetric.
        - range: [min(x), max(x)]
        - https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver
        """
        super(MinMaxQuantizer, self).__init__(org_weight, args)

        if self.per_channel == True:
            _min = org_weight.view(org_weight.size(0), -1).min(dim=1).values
            _max = org_weight.view(org_weight.size(0), -1).max(dim=1).values
        else:
            _min = org_weight.min()
            _max = org_weight.max()

        if self.signed == True:
            # if s8, scaler = (_max - _min) / (127 - (-128))
            # if s8, zero_point = -_min / scaler + (-128)
            self.compute_qparams(_min, _max)
        else:
            # if u8, scaler = (_max - _min) / (255 - 0)
            # if u8, zero_point = -_min / scaler + 0
            self.compute_qparams(_min, _max)


class NormQuantizer(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """
        [ NormQuantizer ]
        - Consider the L2 distance between the original weight and the quantized weight.
        - The [Min, Max] range for computing Qparams are determined by the L2 distance Score.
        - The [Min, Max] combination candicates are smallest p% and largest p% of the sorted weight. respectively.
            - I think that considering about 0.01% ~ 0.00001% is enough. (inspired by the percentile method)
        """
        super(NormQuantizer, self).__init__(org_weight, args)

        if self.per_channel == True:
            _argsorted = torch.argsort(org_weight.view(org_weight.size(0), -1))
            best_min = org_weight.view(org_weight.size(0), -1)[
                torch.arange(org_weight.size(0)), _argsorted[:, 0]
            ]
            best_max = org_weight.view(org_weight.size(0), -1)[
                torch.arange(org_weight.size(0)), _argsorted[:, -1]
            ]

        else:
            _argsorted = torch.argsort(org_weight.view(-1))
            best_min = org_weight.view(-1)[_argsorted[0]].clone()
            best_max = org_weight.view(-1)[_argsorted[-1]].clone()

        # Default p is 2.4
        # L_p norm minimization as described in LAPQ
        # https://arxiv.org/abs/1911.07190
        self._p = args.get("p") if args.get("p") else 2.4
        print(f"    p = {self._p}")

        def forward_copy(input: Tensor) -> Tensor:
            # Avoid override errors when using AdaRound's self.forward function to perform the initialization process.
            return self._dequantize(
                torch.clamp(
                    (input / self._scaler).round() + self._zero_point,
                    self._repr_min,
                    self._repr_max,
                )
            )

        if self.signed == True:
            # perform_2D_search [min, max]
            self.compute_qparams(best_min, best_max)
            # best_score = (self.forward(org_weight) - org_weight).norm(p=self._p)
            best_score = (forward_copy(org_weight) - org_weight).norm(p=self._p)

            _iter_len = int((len(_argsorted / 2) + 1) * 0.001) + 10
            # the combination of smallest 0.05% and largest 0.05% is enough.
            # if compute all combination, there are huge overhead without significant improvement.
            for _idx in range(1, _iter_len):
                for i, j in {(0, 1), (1, 0), (1, 1)}:
                    # Consider the combination with +- 1 index value
                    # init : (0, -1)
                    # loop : (0, -2), (1, -1), (1, -2)
                    # next : (1, -3), (2, -2), (2, -3).. not duplicated.
                    lowidx = _idx - 1 + i
                    highidx = -_idx - j

                    if self.per_channel == True:
                        _tmp_min = org_weight.view(org_weight.size(0), -1)[
                            torch.arange(org_weight.size(0)),
                            _argsorted[:, lowidx],
                        ]
                        _tmp_max = org_weight.view(org_weight.size(0), -1)[
                            torch.arange(org_weight.size(0)), _argsorted[:, highidx]
                        ]
                    else:
                        _tmp_min = org_weight.view(-1)[_argsorted[lowidx]]
                        _tmp_max = org_weight.view(-1)[_argsorted[highidx]]

                    self.compute_qparams(_tmp_min, _tmp_max)
                    # -> Changed the scaler and zero_point

                    # _tmp_score = (self.forward(org_weight) - org_weight).norm(p=self._p)
                    _tmp_score = (forward_copy(org_weight) - org_weight).norm(p=self._p)

                    best_min = torch.where(_tmp_score < best_score, _tmp_min, best_min)
                    best_max = torch.where(_tmp_score < best_score, _tmp_max, best_max)
                    best_score = torch.min(_tmp_score, best_score)

            # -> Findinf the best_min, best_max is done..
            self.compute_qparams(best_min, best_max)

        else:
            # perform_1D_search [0, max]
            # [ ] perform_1D_search >> NOT YET
            raise NotImplementedError(
                "Unsigned L2DistanceQuantizer is not implemented yet."
            )


class OrgNormQuantizerCode(UniformAffineQuantizer):
    def __init__(self, org_weight, args):
        """ORIGIN SOURCE CODE"""
        super(OrgNormQuantizerCode, self).__init__(org_weight, args)

        if self.signed == True:
            self.channel_wise = self.per_channel

            def lp_loss(pred, tgt, p=2.0):
                x = (pred - tgt).abs().pow(p)
                if not self.channel_wise:
                    return x.mean()
                else:
                    y = torch.flatten(x, 1)
                    return y.mean(1)

            x = org_weight
            if self.channel_wise:
                y = torch.flatten(x, 1)
                x_min, x_max = torch.aminmax(y, dim=1)
                # may also have the one side distribution in some channels
                x_max = torch.max(x_max, torch.zeros_like(x_max))
                x_min = torch.min(x_min, torch.zeros_like(x_min))
            else:
                x_min, x_max = torch.aminmax(x)
            xrange = x_max - x_min
            best_score = torch.zeros_like(x_min) + (1e10)
            best_min = x_min.clone()
            best_max = x_max.clone()
            # enumerate xrange
            self.num = 100

            for i in range(1, self.num + 1):
                tmp_min = torch.zeros_like(x_min)
                tmp_max = xrange / self.num * i
                # tmp_delta = (tmp_max - tmp_min) / (2**self.n_bits - 1)
                tmp_delta = (tmp_max - tmp_min) / (self._repr_max - self._repr_min)
                # enumerate zp
                # for zp in range(0, self.n_levels):
                for zp in range(0, (self._repr_max - self._repr_min + 1)):
                    new_min = tmp_min - zp * tmp_delta
                    new_max = tmp_max - zp * tmp_delta
                    self.compute_qparams(new_max, new_min)
                    # x_q = self.forward(x)
                    x_q = torch.clamp(
                        (x / self._scaler).round() + self._zero_point,
                        self._repr_min,
                        self._repr_max,
                    )
                    x_q = self._dequantize(x_q)

                    score = lp_loss(x, x_q, 2.4)
                    best_min = torch.where(score < best_score, new_min, best_min)
                    best_max = torch.where(score < best_score, new_max, best_max)
                    best_score = torch.min(best_score, score)

            self.compute_qparams(best_min, best_max)
        else:
            # perform_1D_search [0, max]
            ...


def create_AdaRound_Quantizer(scheme, org_weight, args):
    base_classes = {
        "AbsMaxQuantizer": AbsMaxQuantizer,
        "MinMaxQuantizer": MinMaxQuantizer,
        "NormQuantizer": NormQuantizer,
        "OrgNormQuantizerCode": OrgNormQuantizerCode,
    }
    base_class = base_classes.get(scheme)
    if not base_class:
        raise ValueError(f"Unknown base class: {scheme}")

    class AdaRoundQuantizer(base_class):
        def __init__(self, org_weight, args):
            """
            Ref: Up or Down? Adaptive Rounding for Post-Training Quantization
            - https://proceedings.mlr.press/v119/nagel20a/nagel20a.pdf
            """
            super(AdaRoundQuantizer, self).__init__(org_weight, args)
            print(f"    Parent class is {self.__class__.__bases__[0].__name__}")

            self.fp_outputs = None
            # -> Now, We have AbsMaxQuantizer's scaler and zero_point!

            self.zeta = 1.1  # fixed param for function h()
            self.gamma = -0.1  # fixed pamam for function h()
            self.lamda = 1  # lambda. fixed param for regularization function f()

            self._v = None
            self._init_v(org_weight=org_weight)
            self.rouning_value = None

        # [1] init the v value. (h(v) == rounding value)
        def _init_v(self, org_weight: Tensor):
            # [1-1] compute the residual == initial h(v)
            _x_q_round = torch.clamp(
                (org_weight / self._scaler).round() + self._zero_point,
                self._repr_min,
                self._repr_max,
            )
            _x_q_floor = torch.clamp(
                (org_weight / self._scaler).floor() + self._zero_point,
                self._repr_min,
                self._repr_max,
            )

            _residual = _x_q_round - _x_q_floor
            assert torch.all(
                (_residual == 0) | (_residual == 1)
            ), "The residual is {0, 1}."

            # [1-2] compute the v value using inverse h() function
            _v = -torch.log(
                (self.zeta - self.gamma) / (_residual - self.gamma) - 1
            )  # h^-1
            self._v = nn.Parameter(_v, requires_grad=True)
            assert (_residual - self._h()).abs().sum() == 0

            print("    Initiated the V")

        def _h(self) -> Tensor:
            # Rectified_sigmoid (strached sigmoid function)
            return torch.clamp(
                self._v.sigmoid() * (self.zeta - self.gamma) + self.gamma, 0, 1
            )

        def f_reg(self, beta=2.0) -> Tensor:
            # _regularization_term for determining the v
            return (1 - (2 * self._h() - 1).abs().pow(beta)).sum()

        def _quantize(self, input: Tensor) -> Tensor:
            if self.rouning_value == None:
                # return FP
                return torch.clamp(
                    (input / self._scaler).floor() + self._zero_point + self._h(),
                    self._repr_min,
                    self._repr_max,
                )
            else:
                # return INT
                print(",", end="")
                return torch.clamp(
                    (input / self._scaler).floor()
                    + self._zero_point
                    + self.rouning_value,
                    self._repr_min,
                    self._repr_max,
                )

        def setRoundingValues(self):
            FIXED_ROUNDDING_VALUE = self._h().clone().detach()
            self.rouning_value = FIXED_ROUNDDING_VALUE

            assert torch.all(
                (self.rouning_value == 0) | (self.rouning_value == 1)
            ), "The rounding value have to be {0, 1}."

    return AdaRoundQuantizer(org_weight, args)


# Origin Bn fonlding function (2)
def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


# Origin Bn fonlding function (1)
def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data**2


quantizerDict = {
    "AbsMaxQuantizer": AbsMaxQuantizer,
    "MinMaxQuantizer": MinMaxQuantizer,
    "NormQuantizer": NormQuantizer,
    "OrgNormQuantizerCode": OrgNormQuantizerCode,
}


class StraightThrough(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, input):
        return input


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
        if self.a_quant_inited == True and self.a_quant_enable == True:
            # print("A", end="")
            return self.act_quantizer(_Z)
        else:
            return _Z
