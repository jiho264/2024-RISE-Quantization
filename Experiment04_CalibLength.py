import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *
from src.override_resnet import *


###############################################################################################
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


###############################################################################################
"""
activation == HistogramObserver
"""
import numpy as np

cases = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for case in cases:
    per_calib = case / 100 * 5005
    # prepare the model
    _model = resnet50_quan(weights=pretrained_weights_mapping[50])
    _model.to("cpu")
    _model.eval()

    # set fuse ############################################################
    _model = fuse_ALL(_model)
    # set qconfig

    _model.qconfig = torch.quantization.get_default_qconfig("x86")
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
    _, _ = SingleEpochEval(_model, train_loader, criterion, "cuda", limit=per_calib)

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
    # test loader shuffle=True
    eval_loss, eval_acc = SingleEpochEval(
        model=_model, testloader=test_loader, criterion=criterion, device="cpu"
    )
    model_size = get_size_of_model(_model)
    inference_time = run_benchmark(_model, test_loader, "cpu", 10)
    print("------------------------------------------------------------")
    print(f"Case: Use calib {case}%")
    print(f"Model Size: {model_size}MB")
    print(f"Inference Time: {inference_time:.2f}ms")
    print(f"Eval Loss: {eval_loss:.4f}")
    print(f"Eval Acc: {eval_acc:.3f}%")
    print("\n")

print("Done!")
