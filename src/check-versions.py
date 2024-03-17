import torch
import sys
import subprocess


print("")
print("- Ubuntu : ")
# Execute the command in a shell
subprocess.call("uname -a", shell=True)

print("")
print("- Python : ")
print("Python version :", sys.version)

print("")
print("- CUDA :")
subprocess.call("nvcc -V", shell=True)
print("cuda available on python: ", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


print("")
print("- PyTorch : ")
print("torch version : ", torch.__version__)

print("")
print("- cuDNN :")
# subprocess.call(
#     "cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2", shell=True
# )
print("cudnn ver : ", torch.backends.cudnn.version())
print("cudnn enabled:", torch.backends.cudnn.enabled)

print("")
import tensorrt

print("- TensorRT : ")
subprocess.call("dpkg-query -W tensorrt", shell=True)
print(tensorrt.__version__)
print("")
# os="ubuntu2204"
# tag="8.6.1-cuda-12.0"
# sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
# sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
