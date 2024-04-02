import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
    check_min_max_valid,
    calculate_qmin_qmax,
    is_per_tensor,
    is_per_channel,
    validate_qmin_qmax,
)

from torch.quantization.observer import HistogramObserver


class CustomHistogramObserver(HistogramObserver):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                        used to interpolate histograms with varying ranges across observations
        dtype: dtype argument to the `quantize` node needed to implement the
                reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """

    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range: bool = False,
        eps: float = torch.finfo(torch.float32).eps,
    ) -> None:
        super(CustomHistogramObserver, self).__init__(
            bins=bins,
            upsample_rate=upsample_rate,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            eps=eps,
        )
