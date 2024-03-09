import torch
import sys

print("")
print("Python version :", sys.version)
print("torch version : ", torch.__version__)
print("cuda available : ", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("cudnn ver : ", torch.backends.cudnn.version())
print("cudnn enabled:", torch.backends.cudnn.enabled)
print("")
