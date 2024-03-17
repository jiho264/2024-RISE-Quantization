import os


def print_environment_variables():
    for key, value in os.environ.items():
        if "PATH" in key or "LD_LIBRARY_PATH" in key:
            print(f"{key}: {value}")


print_environment_variables()

import tensorrt

# import torch
# import sys
# import subprocess

print(tensorrt.__version__)

# LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64:/usr/local/cuda/lib64::/usr/local/cuda-12.1/lib64

# PATH: /usr/local/cuda/bin:/home/lee/miniconda3/envs/py312/bin:/home/lee/.vscode-server/bin/863d2581ecda6849923a2118d93a088b0745d9d6/bin/remote-cli:/usr/local/cuda-12.1/bin:/usr/local/cuda/bin:/home/lee/miniconda3/condabin:/home/lee/miniconda3/envs/py312/bin:/home/lee/miniconda3/envs/py312/bin:/home/lee/.vscode-server/bin/863d2581ecda6849923a2118d93a088b0745d9d6/bin/remote-cli:/home/lee/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-12.1/bin


# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# python3 -m pip install --upgrade tensorrt
# python3 -m pip install --upgrade tensorrt_lean
# python3 -m pip install --upgrade tensorrt_dispatch
