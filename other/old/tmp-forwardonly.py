import torch, time, copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.quantization import prepare, convert
from src.utils import *
from src.override_resnet import *


model = resnet50_quan(weights=pretrained_weights_mapping[50])
model.to("cuda")
model.eval()

# set fuse ############################################################
model = fuse_ALL(model)

model.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.HistogramObserver.with_args(reduce_range=True),
    weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
)
prepare(model, inplace=True)

# calibrate the model ############################################################
criterion = nn.CrossEntropyLoss()
train_loader, _ = GetDataset(
    dataset_name="ImageNet",
    device="cuda",
    root="data",
    batch_size=32,
    num_workers=8,
)
print(SingleEpochEval(model, train_loader, criterion, "cuda", 1))

# for name, module in model.named_modules():
#     print(name, module)


# %%
for name, module in model.named_modules():
    if (
        len(name) > 2
        and name[-7:] == "through"
        and hasattr(module, "activation_post_process")
    ):
        # if name == "act_obs":
        print(f"{name}", end=" ")

        _hist = list()
        print(module.activation_post_process.histogram.shape)
        # for x in module.activation_post_process.get_tensor_value():
        #     _hist.append(torch.tensor(x).to("cpu"))

        #     print(torch.tensor(x).to("cpu").numpy().shape, end=" ")

        # _hist = np.array(_hist).flatten()

        # cnt = 1
        # for i in _hist.shape:
        #     cnt *= i

        # tmp = cnt * 4 / 1024 / 1024, "MB"
        # print(tmp)
