import torch, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *
from src.override_resnet import *


from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


device = "cpu"
_model = resnet50_quan(weights=pretrained_weights_mapping[50])
_model.eval()

# set fuse ############################################################
_model = fuse_ALL(_model)


example_inputs = torch.randn(1, 3, 224, 224)
qconfig_mapping = get_default_qconfig_mapping("fbgemm")

_model = prepare_fx(_model, qconfig_mapping, example_inputs)


# calibrate the model ############################################################
criterion = nn.CrossEntropyLoss()
train_loader, _ = GetDataset(
    dataset_name="ImageNet",
    device=device,
    root="data",
    batch_size=32,
    num_workers=8,
)
_, _ = SingleEpochEval(_model, train_loader, criterion, device, 4)

# convert the model ############################################################

_model = convert_fx(_model)

# evaluate the model ############################################################

_batch_size = 32
_, test_loader = GetDataset(
    dataset_name="ImageNet",
    device=device,
    root="data",
    batch_size=_batch_size,
    num_workers=8,
)

eval_loss, eval_acc = SingleEpochEval(
    model=_model,
    testloader=test_loader,
    criterion=criterion,
    device=device,
    limit=10,
)
model_size = get_size_of_model(_model)
inference_time = run_benchmark(_model, test_loader, device, 10)
print("------------------------------------------------------------")
# print(f"case_activation: {case_activation}")
# print(f"case_upsample_rate: {i}")
print(f"Model Size: {model_size:.2f}MB")
print(f"Inference Time: {inference_time:.2f}ms")
print(f"Eval Loss: {eval_loss:.4f}")
print(f"Eval Acc: {eval_acc:.3f}%")
print("\n")

print("Done!")
