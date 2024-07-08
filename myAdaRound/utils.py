import torch, tqdm
from torch import Tensor


#################################################################################################
## 1. Prepare the dataset and utility functions
#################################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def GetDataset(batch_size=64):
    import torchvision
    import torchvision.transforms as transforms

    train_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    test_dataset = torchvision.datasets.ImageNet(
        root="data/ImageNet",
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, device):
    model.eval().to(device)
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    cnt = 0
    with torch.no_grad():
        # for image, target in tqdm.tqdm(data_loader):
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            # loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


#################################################################################################
#################################################################################################
#################################################################################################

import torch.nn as nn
import torch.nn.functional as F


class QuantModule(nn.Module):
    def __init__(self, org_module):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.clone()

        """quantizer"""
        self.scaler = 0
        self.zero_point = 0
        self.alpha = 0
        self.theta = 0

    def quantize(self, input: Tensor) -> Tensor:
        return torch.clamp((input / self.scaler).round() + self.zero_point, 0, 255)

    def dequantize(self, input: Tensor) -> Tensor:
        return (input - self.zero_point) * self.scaler

    def forward(self, x: Tensor) -> Tensor:
        x = self.dequantize(self.quantize(x))
        return self.fwd_func(x, self.weight, **self.fwd_kwargs)

        """
        ResNet18 quantization with myAdaRound...
        QuantModule: conv1, torch.Size([64, 3, 7, 7])
        QuantModule: layer1.0.conv1, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.0.conv2, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.1.conv1, torch.Size([64, 64, 3, 3])
        QuantModule: layer1.1.conv2, torch.Size([64, 64, 3, 3])
        QuantModule: layer2.0.conv1, torch.Size([128, 64, 3, 3])
        QuantModule: layer2.0.conv2, torch.Size([128, 128, 3, 3])
        QuantModule: layer2.0.downsample.0, torch.Size([128, 64, 1, 1])
        QuantModule: layer2.1.conv1, torch.Size([128, 128, 3, 3])
        QuantModule: layer2.1.conv2, torch.Size([128, 128, 3, 3])
        QuantModule: layer3.0.conv1, torch.Size([256, 128, 3, 3])
        QuantModule: layer3.0.conv2, torch.Size([256, 256, 3, 3])
        QuantModule: layer3.0.downsample.0, torch.Size([256, 128, 1, 1])
        QuantModule: layer3.1.conv1, torch.Size([256, 256, 3, 3])
        QuantModule: layer3.1.conv2, torch.Size([256, 256, 3, 3])
        QuantModule: layer4.0.conv1, torch.Size([512, 256, 3, 3])
        QuantModule: layer4.0.conv2, torch.Size([512, 512, 3, 3])
        QuantModule: layer4.0.downsample.0, torch.Size([512, 256, 1, 1])
        QuantModule: layer4.1.conv1, torch.Size([512, 512, 3, 3])
        QuantModule: layer4.1.conv2, torch.Size([512, 512, 3, 3])
        QuantModule: fc, torch.Size([1000, 512])
        """
