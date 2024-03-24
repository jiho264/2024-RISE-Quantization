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
    torch.quantization.HistogramObserver.with_args(dtype=torch.qint8),
    torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
    torch.quantization.MovingAverageMinMaxObserver.with_args(dtype=torch.qint8),
    torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
    torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8
    ),
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

for case_activation in cases_activation:
    for case_weight in cases_weight:
        # prepare the model
        _model = resnet50_quan(weights=pretrained_weights_mapping[50])
        _model.to("cpu")
        _model.eval()

        # set fuse ############################################################
        _model = fuse_ALL(_model)
        # set qconfig
        _model.qconfig = torch.quantization.QConfig(
            activation=case_activation,
            weight=case_weight,
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
        _, _ = SingleEpochEval(_model, train_loader, criterion, "cuda")

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
            model=_model, testloader=test_loader, criterion=criterion, device="cpu"
        )
        model_size = get_size_of_model(_model)
        inference_time = run_benchmark(_model, test_loader, "cpu", 10)
        print("------------------------------------------------------------")
        print(f"case_activation: {case_activation}")
        print(f"case_weight: {case_weight}")
        print(f"Model Size: {model_size:.2f}MB")
        print(f"Inference Time: {inference_time:.2f}ms")
        print(f"Eval Loss: {eval_loss:.4f}")
        print(f"Eval Acc: {eval_acc:.3f}%")
        print("\n")

print("Done!")
