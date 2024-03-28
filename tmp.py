import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *
from src.override_resnet import *


"""
torch.ao.quantization.qconfig.py

def get_default_qconfig(backend='x86', version=0):
    # Returns the default PTQ qconfig for the specified backend.

    # Args:
    #   * `backend` (str): a string representing the target backend. Currently supports
    #     `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    # Return:
    #     qconfig

    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: " + str(backend) +
            f" not supported. backend must be one of {supported_backends}"
        )

    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                                weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                                weight=default_weight_observer)
        elif backend == 'onednn':
            if not torch.cpu._is_cpu_support_vnni():
                warnings.warn(
                    "Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues "
                    "on CPU without Vector Neural Network Instruction support.")
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                                weight=default_per_channel_weight_observer)
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                                weight=default_per_channel_weight_observer)
        else:
            # won't reach
            qconfig = default_qconfig
    else:
        raise AssertionError("Version number: " + str(version) +
                                " in get_default_qconfig is not supported. Version number must be 0")

    return qconfig
    
"""
# All default activation observer is HistogramObserver
cases_activation = [
    torch.quantization.HistogramObserver.with_args(reduce_range=True),
    torch.quantization.HistogramObserver.with_args(reduce_range=False),
]
# we can use 5 different type of weight observer
cases_weight = [
    # torch.quantization.HistogramObserver.with_args(dtype=torch.qint8),
    # torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
    # torch.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
    torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
    # torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8),
]


def fuse_ALL(model) -> nn.Module:
    SingleTimeFlag = False
    for m in model.modules():
        if m.__class__.__name__ == ResNet_quan.__name__:
            if SingleTimeFlag == True:
                raise ValueError("ResNet_quan is already fused")
            SingleTimeFlag = True
            torch.quantization.fuse_modules(
                m,
                ["conv1", "bn1", "relu"],
                inplace=True,
            )

        if type(m) == BottleNeck_quan:

            torch.quantization.fuse_modules(
                m,
                [
                    ["conv1", "bn1", "relu1"],
                    ["conv2", "bn2", "relu2"],
                    ["conv3", "bn3"],
                ],
                inplace=True,
            )
            if m.downsample is not None:
                torch.quantization.fuse_modules(
                    m.downsample,
                    ["0", "1"],
                    inplace=True,
                )
    return model


# for case_activation in cases_activation:
#     for case_weight in cases_weight:
# prepare the model
# for i in [120, 121, 122, 123, 124, 125, 126, 127, 128]:
for i in range(110, 121):
    _model = resnet50_quan(weights=pretrained_weights_mapping[50])
    _model.to("cpu")
    _model.eval()

    # set fuse ############################################################
    _model = fuse_ALL(_model)

    _model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.HistogramObserver.with_args(
            quant_min=0, quant_max=127, upsample_rate=128
        ),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
    )
    prepare(_model, inplace=True)

    # calibrate the model ############################################################
    criterion = nn.CrossEntropyLoss()
    train_loader, _ = GetDataset(
        dataset_name="ImageNet",
        device="cuda",
        root="data",
        batch_size=256,
        num_workers=8,
    )
    _, _ = SingleEpochEval(_model, train_loader, criterion, "cuda", 4)

    # convert the model ############################################################
    _model.to("cpu")
    convert(_model, inplace=True)

    # evaluate the model ############################################################

    _batch_size = 32
    _, test_loader = GetDataset(
        dataset_name="ImageNet",
        device="cpu",
        root="data",
        batch_size=_batch_size,
        num_workers=8,
    )

    eval_loss, eval_acc = SingleEpochEval(
        model=_model,
        testloader=test_loader,
        criterion=criterion,
        device="cpu",
        limit=10,
    )
    model_size = get_size_of_model(_model)
    inference_time = run_benchmark(_model, test_loader, "cpu", 10)
    print("------------------------------------------------------------")
    # print(f"case_activation: {case_activation}")
    # print(f"case_upsample_rate: {i}")
    print(f"Model Size: {model_size:.2f}MB")
    print(f"Inference Time: {inference_time:.2f}ms")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Acc: {eval_acc:.3f}%")
    print("\n")

print("Done!")


"""

UniformQuantizationObserverBase(ObservingBase):
    def _calculate_qparams(self, min_val, max_val):
        ...
        
        
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        ## min_val_neg == 0
        ## max_val_pos == max_val
        
        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            ...
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            ...
        else:
        
            # Activation의 quantization은 per_tensor이므로 항상 여기서 계산됨.
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
"""
