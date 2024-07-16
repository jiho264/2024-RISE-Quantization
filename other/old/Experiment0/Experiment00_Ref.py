import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *


# prepare the model
_model = resnet50(weights=pretrained_weights_mapping[50])
_model.to("cuda")
_model.eval()
# calibrate the model ############################################################
criterion = nn.CrossEntropyLoss()
_, test_loader = GetDataset(
    dataset_name="ImageNet",
    device="cuda",
    root="data",
    batch_size=256,
    num_workers=8,
)
eval_loss, eval_acc = SingleEpochEval(_model, test_loader, criterion, "cuda")


model_size = get_size_of_model(_model)
_batch_size = 32
_, test_loader = GetDataset(
    dataset_name="ImageNet",
    device="cpu",
    root="data",
    batch_size=_batch_size,
    num_workers=8,
)
_model.to("cpu")
inference_time = run_benchmark(_model, test_loader, "cpu", 10)
print("------------------------------------------------------------")
print(f"Case: Reference")
print(f"Model Size: {model_size:.2f}MB")
print(f"Inference Time: {inference_time:.2f}ms")
print(f"Eval Loss: {eval_loss:.4f}")
print(f"Eval Acc: {eval_acc:.3f}%")
print("\n")

print("Done!")
